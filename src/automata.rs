//! NFA and DFA
//!
//! This module implements NFA (Nondeterministic finite automaton) which is can be combined
//! in more complicated one with Thompson's construction and complied to DFA (Deterministic
//! finaite automaton) with power set construction.
use std::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    fmt,
    iter::once,
    rc::Rc,
};

type Symbol = u8;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NFAStateId(usize);

type NFAStateSet = Rc<BTreeSet<NFAStateId>>;

#[derive(Clone)]
struct NFAState<T> {
    edges: BTreeMap<Symbol, NFAStateId>,
    epsilons: BTreeSet<NFAStateId>,
    tag: Option<T>,
}

impl<T> NFAState<T> {
    fn new() -> Self {
        Self {
            edges: Default::default(),
            epsilons: Default::default(),
            tag: None,
        }
    }
}

#[derive(Clone)]
pub struct NFA<T> {
    start: NFAStateId,
    stop: NFAStateId,
    states: BTreeMap<NFAStateId, NFAState<T>>,
}

/// Nondeterministic finite automaton
///
/// References:
///   - [Regular Expression Matching Can Be Simple And Fast](https://swtch.com/~rsc/regexp/regexp1.html)
impl<T> NFA<T> {
    // Assign provided tag to the stop state
    pub fn tag(mut self, tag: impl Into<T>) -> Self {
        if let Some(state) = self.states.get_mut(&self.stop) {
            state.tag = Some(tag.into());
        }
        self
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> NFA<U> {
        let Self {
            start,
            stop,
            states,
        } = self;
        let states = states
            .into_iter()
            .map(|(state_id, state)| {
                let NFAState {
                    edges,
                    epsilons,
                    tag,
                } = state;
                let state = NFAState {
                    edges,
                    epsilons,
                    tag: tag.map(&mut f),
                };
                (state_id, state)
            })
            .collect();
        NFA {
            start,
            stop,
            states,
        }
    }

    /// Number of states inside NFA
    pub fn size(&self) -> usize {
        self.states.len()
    }

    /// Create NFA that matches single symbol from alphabet given predicate
    pub fn predicate(pred: impl Fn(Symbol) -> bool) -> Self {
        let start = NFAStateId(0);
        let stop = NFAStateId(1);
        let mut state = NFAState::new();
        for symbol in 0..=Symbol::max_value() {
            if pred(symbol) {
                state.edges.insert(symbol, stop);
            }
        }

        let mut states = BTreeMap::new();
        states.insert(start, state);
        states.insert(stop, NFAState::new());

        Self {
            start,
            stop,
            states,
        }
    }

    /// Empty NFA
    ///
    /// Contains single state, which is start and stop at the same time
    /// with no edges.
    pub fn empty() -> Self {
        let state_id = NFAStateId(0);
        let mut states = BTreeMap::new();
        states.insert(state_id, NFAState::new());
        Self {
            start: state_id,
            stop: state_id,
            states,
        }
    }

    /// NFA that matches nothing
    ///
    /// Two disconnected states start and stop.
    pub fn nothing() -> Self {
        let start = NFAStateId(0);
        let stop = NFAStateId(1);
        let mut states = BTreeMap::new();
        states.insert(start, NFAState::new());
        states.insert(stop, NFAState::new());
        Self {
            start,
            stop,
            states,
        }
    }

    /// Thompson's construction of NFAs chain one after another.
    /// For `a` and `b` regular expressions it is equivalent to `ab` expressions.
    pub fn sequence(nfas: impl IntoIterator<Item = Self>) -> Self {
        let (mut states, ends) = Self::merge_states(nfas, 0);
        if ends.is_empty() {
            return Self::empty();
        }

        // connect stops from current NFA to the starts of the next NFA
        // with epsilon edge.
        for index in 1..ends.len() {
            let (_, from) = ends[index - 1];
            let (to, _) = ends[index];
            if let Some(from_state) = states.get_mut(&from) {
                from_state.epsilons.insert(to);
            }
        }

        let (start, _) = ends[0];
        let (_, stop) = ends[ends.len() - 1];

        Self {
            start,
            stop,
            states,
        }
    }

    /// Thompson's construction of NFA that matches any of the provided NFAs
    /// For `a` and `b` regular expressions it is equivalent to `(a|b)` expressions.
    pub fn choice(nfas: impl IntoIterator<Item = Self>) -> Self {
        let (mut states, ends) = Self::merge_states(nfas, 2);
        if ends.is_empty() {
            return Self::nothing();
        }

        let start = NFAStateId(0);
        let stop = NFAStateId(1);
        let mut start_state = NFAState::new();
        for (from, to) in ends {
            start_state.epsilons.insert(from);
            if let Some(to_state) = states.get_mut(&to) {
                to_state.epsilons.insert(stop);
            }
        }
        states.insert(start, start_state);
        states.insert(stop, NFAState::new());

        Self {
            start,
            stop,
            states,
        }
    }

    /// For `a` regular expression it is equivalent to `a+`
    pub fn some(mut self) -> Self {
        if let Some(stop) = self.states.get_mut(&self.stop) {
            stop.epsilons.insert(self.start);
        }
        self
    }

    pub fn optional(mut self) -> Self {
        if let Some(start) = self.states.get_mut(&self.start) {
            start.epsilons.insert(self.stop);
        }
        self
    }

    /// For `a` regular expression it is equivalent to `a*`
    pub fn many(self) -> Self {
        // add offset of 2 to state ids
        let (mut states, ends) = Self::merge_states(once(self), 2);
        let (from, to) = ends[0];

        let start = NFAStateId(0);
        let stop = NFAStateId(1);
        let mut start_state = NFAState::new();
        start_state.epsilons.insert(from);
        start_state.epsilons.insert(stop);
        if let Some(to_state) = states.get_mut(&to) {
            to_state.epsilons.insert(stop);
            to_state.epsilons.insert(from);
        }
        states.insert(start, start_state);
        states.insert(stop, NFAState::new());

        Self {
            start,
            stop,
            states,
        }
    }

    /// Merge multiple NFAs
    ///
    /// - recalulate ids so ids from differnt NFAs would be different
    /// - merges states into a single mapping
    /// - merges stops and starts preserving order
    fn merge_states(
        nfas: impl IntoIterator<Item = Self>,
        mut offset: usize,
    ) -> (
        // states
        BTreeMap<NFAStateId, NFAState<T>>,
        // starts/stops
        Vec<(NFAStateId, NFAStateId)>,
    ) {
        let mut states_out: BTreeMap<NFAStateId, NFAState<T>> = BTreeMap::new();
        let mut ends_out: Vec<(NFAStateId, NFAStateId)> = Vec::new();

        for NFA {
            start,
            stop,
            states,
        } in nfas
        {
            let start = NFAStateId(offset + start.0);
            let stop = NFAStateId(offset + stop.0);
            ends_out.push((start, stop));

            let mut max_id = 0;
            for (id, state) in states {
                max_id = std::cmp::max(max_id, id.0);
                let NFAState {
                    edges,
                    epsilons,
                    tag,
                } = state;
                let id = NFAStateId(offset + id.0);
                let edges = edges
                    .into_iter()
                    .map(|(k, v)| (k, NFAStateId(offset + v.0)))
                    .collect();
                let epsilons = epsilons
                    .into_iter()
                    .map(|v| NFAStateId(offset + v.0))
                    .collect();
                states_out.insert(
                    id,
                    NFAState {
                        edges,
                        epsilons,
                        tag,
                    },
                );
            }

            offset += max_id + 1;
        }
        (states_out, ends_out)
    }

    /// NFA to DFA using powerset construction
    pub fn compile(&self) -> DFA<T>
    where
        T: Clone + Ord,
    {
        // each DFA state is represented as epsilon closure of NFA states `Rc<BTreeSet<NFAStateId>>`
        let mut dfa_states: BTreeMap<NFAStateSet, DFAState> = BTreeMap::new();
        // constructed DFA
        let mut dfa_table: BTreeMap<DFAState, BTreeMap<Symbol, DFAState>> = BTreeMap::new();

        // initialize traversal queue with initial state
        let dfa_start = self.epsilon_closure(once(self.start));
        let dfa_start_id = DFAState(0);
        let mut dfa_queue = vec![(dfa_start_id, dfa_start.clone())]; // to be traversed DFA states
        dfa_states.insert(dfa_start, DFAState(0));

        while let Some((dfa_state_id, dfa_state)) = dfa_queue.pop() {
            // find all unique symbols leading from the current DFA state
            let symbols: BTreeSet<Symbol> = dfa_state
                .iter()
                .flat_map(|nfa_state_id| self.states[nfa_state_id].edges.keys().copied())
                .collect();

            // resolve all edges of the current DFA state
            let mut dfa_edges: BTreeMap<Symbol, DFAState> = BTreeMap::new();
            for symbol in symbols {
                // calculate new DFA state for a given symbol
                let dfa_state_new = dfa_state
                    .iter()
                    .flat_map(|nfa_state_id| self.states[nfa_state_id].edges.get(&symbol).copied());
                let dfa_state_new = self.epsilon_closure(dfa_state_new);
                // enqueue state and allocate id if it has not been found yet
                let dfa_state_new_id = match dfa_states.get(&dfa_state_new) {
                    Some(id) => *id,
                    None => {
                        let id = DFAState(dfa_states.len());
                        dfa_states.insert(dfa_state_new.clone(), id);
                        dfa_queue.push((id, dfa_state_new));
                        id
                    }
                };
                // update edges of the current DFA state
                dfa_edges.insert(symbol, dfa_state_new_id);
            }
            // update DFA table
            dfa_table.insert(dfa_state_id, dfa_edges);
        }

        // construct state information
        let mut infos: Vec<DFAStateInfo<T>> = Vec::new();
        infos.resize_with(dfa_states.len(), || DFAStateInfo {
            accepting: false,
            terminal: false,
            tags: Default::default(),
        });
        for (dfa_state, dfa_state_id) in dfa_states {
            let info = &mut infos[dfa_state_id.0];
            info.accepting = dfa_state.contains(&self.stop);
            info.terminal = dfa_table[&dfa_state_id].is_empty();
            for nfa_state_id in dfa_state.iter() {
                if let Some(tag) = self.states.get(nfa_state_id).and_then(|s| s.tag.clone()) {
                    info.tags.insert(tag.clone());
                }
            }
        }

        // by construction all states are dense (meaning they start from 0 and do not have gaps)
        let lang_size = Symbol::max_value() as usize + 1;
        let states = dfa_table
            .into_iter()
            .enumerate()
            .flat_map(|(index, (state, edges))| {
                assert_eq!(index, state.0);
                (0..=Symbol::max_value()).map(move |symbol| edges.get(&symbol).copied())
            })
            .collect::<Vec<Option<DFAState>>>();

        DFA {
            start: dfa_start_id,
            states: states.into_boxed_slice(),
            infos: infos.into_boxed_slice(),
            lang_size,
        }
    }

    /// All states reachable from provided set of states with epsilon transition
    fn epsilon_closure(&self, states: impl IntoIterator<Item = NFAStateId>) -> NFAStateSet {
        let mut output = BTreeSet::new();
        let mut queue: Vec<_> = states.into_iter().collect();
        while let Some(state_id) = queue.pop() {
            let state = &self.states[&state_id];
            for epsilon_id in state.epsilons.iter() {
                if output.contains(epsilon_id) {
                    continue;
                }
                queue.push(*epsilon_id);
            }
            output.insert(state_id);
        }
        Rc::new(output)
    }

    /// Create NFA that matches single digit 0..9
    pub fn digit() -> Self {
        Self::predicate(|symbol| symbol.is_ascii_digit())
    }

    /// Create NFA that matches positive integer number
    pub fn number() -> Self {
        Self::digit().some()
    }
}

impl<'a, T> From<&'a str> for NFA<T> {
    fn from(string: &'a str) -> Self {
        let start = NFAStateId(0);

        let mut state_id = start;
        let mut state = NFAState::new();
        let mut states = BTreeMap::new();
        for (index, symbol) in string.bytes().enumerate() {
            let next_id = NFAStateId(index + 1);
            let next_state = NFAState::new();
            state.edges.insert(symbol, next_id);
            states.insert(state_id, std::mem::replace(&mut state, next_state));
            state_id = next_id;
        }
        states.insert(state_id, state);

        Self {
            start,
            stop: state_id,
            states,
        }
    }
}

impl<T> std::ops::BitOr<NFA<T>> for NFA<T> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self::choice(once(self).chain(once(rhs)))
    }
}

impl<T> std::ops::Add<NFA<T>> for NFA<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::sequence(once(self).chain(once(rhs)))
    }
}

impl<T> fmt::Debug for NFA<T>
where
    T: fmt::Debug,
{
    /// Format NFA as a valid DOT graph
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\ndigraph NFA {{")?;
        writeln!(f, "  rankdir=\"LR\"")?;

        for (from, state) in self.states.iter() {
            // node
            write!(f, "  {} [", from.0)?;
            if from == &self.stop {
                write!(f, "shape=doublecircle")?;
            } else {
                write!(f, "shape=circle")?;
            }
            if let Some(tag) = &state.tag {
                write!(f, ",label=\"{} {{{:?}}}\"", from.0, tag)?;
            }
            writeln!(f, "]")?;

            // edges
            for (symbol, to) in state.edges.iter() {
                writeln!(
                    f,
                    "  {} -> {} [label=\"{}\"]",
                    from.0,
                    to.0,
                    char::from(*symbol).escape_default(),
                )?;
            }
            for to in state.epsilons.iter() {
                writeln!(f, "  {} -> {} [color=red]", from.0, to.0)?;
            }
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DFAState(usize);

pub struct DFAStateInfo<T> {
    /// Input can be accepted at this point
    pub accepting: bool,
    /// There is no outgoing edges from this state
    pub terminal: bool,
    /// All tags associated with this node
    pub tags: BTreeSet<T>,
}

pub struct DFA<T> {
    start: DFAState,
    states: Box<[Option<DFAState>]>,
    infos: Box<[DFAStateInfo<T>]>,
    lang_size: usize,
}

impl<T> DFA<T> {
    /// Nuber of states in DFA
    pub fn size(&self) -> usize {
        self.states.len() / self.lang_size
    }

    /// Get start state of the DFA
    pub fn start(&self) -> DFAState {
        self.start
    }

    /// Get information about DFA state
    pub fn info(&self, state: DFAState) -> &DFAStateInfo<T> {
        &self.infos[state.0]
    }

    /// Transition from provided state given symbol
    pub fn transition(&self, state: DFAState, symbol: Symbol) -> Option<DFAState> {
        self.states[self.lang_size * state.0 + symbol as usize]
    }

    pub fn transition_many(
        &self,
        state: DFAState,
        symbols: impl IntoIterator<Item = Symbol>,
    ) -> Option<DFAState> {
        symbols
            .into_iter()
            .try_fold(state, |state, symbol| self.transition(state, symbol))
    }

    /// Check if DFA accepts given input
    pub fn matches(&self, symbols: impl IntoIterator<Item = Symbol>) -> bool {
        if let Some(state) = self.transition_many(self.start(), symbols) {
            self.info(state).accepting
        } else {
            false
        }
    }
}

impl<T> fmt::Debug for DFA<T>
where
    T: fmt::Debug,
{
    /// Format DFA as a valid DOT graph
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\ndigraph DFA {{")?;
        writeln!(f, "  rankdir=\"LR\"")?;

        for from in (0..self.size()).map(DFAState) {
            // node
            let info = self.info(from);
            write!(f, "  {} [", from.0)?;
            if info.accepting {
                write!(f, "shape=doublecircle")?;
            } else {
                write!(f, "shape=circle")?;
            }
            if info.terminal {
                write!(f, ",color=red")?
            }
            if !info.tags.is_empty() {
                write!(f, ",label=\"{} {:?}\"", from.0, &info.tags)?;
            }
            writeln!(f, "]")?;

            // edges
            for symbol in 0..self.lang_size {
                let symbol = symbol as Symbol;
                match self.transition(from, symbol) {
                    None => continue,
                    Some(to) => {
                        writeln!(
                            f,
                            "  {} -> {} [label=\"{}\"]",
                            from.0,
                            to.0,
                            char::from(symbol).escape_default(),
                        )?;
                    }
                }
            }
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfa_simple() {
        let nfa = NFA::from("abc").tag(1) | NFA::from("abd").tag(2);
        let dfa = nfa.compile();

        assert_eq!(dfa.size(), 5);
        assert!(dfa.matches("abc".bytes()));
        assert!(dfa.matches("abd".bytes()));
        assert!(!dfa.matches("abe".bytes()));

        let state = dfa
            .transition_many(dfa.start(), "abc".bytes())
            .expect("should match");
        assert_eq!(&dfa.info(state).tags, &once(1).collect::<BTreeSet<_>>());

        let state = dfa
            .transition_many(dfa.start(), "abd".bytes())
            .expect("should match");
        assert_eq!(&dfa.info(state).tags, &once(2).collect::<BTreeSet<_>>());
    }

    #[test]
    fn test_nfa_from_str() {
        let nfa: NFA<()> = NFA::from("abc");
        assert_eq!(nfa.size(), 4);
    }

    #[test]
    fn test_digit() {
        let number: DFA<()> = NFA::digit().some().compile();
        assert!(!number.matches("".bytes()));
        assert!(number.matches("127".bytes()));
        assert!(!number.matches("13a".bytes()));
    }
}
