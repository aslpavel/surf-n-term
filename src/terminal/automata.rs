use std::{
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
struct NFAState {
    edges: BTreeMap<Symbol, NFAStateId>,
    epsilons: BTreeSet<NFAStateId>,
}

impl NFAState {
    fn new() -> Self {
        Self {
            edges: Default::default(),
            epsilons: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct NFA<T> {
    start: NFAStateId,
    stop: NFAStateId,
    tags: BTreeMap<NFAStateId, T>,
    states: BTreeMap<NFAStateId, NFAState>,
}

impl<T> NFA<T> {
    pub fn tag(self, tag: T) -> Self {
        let Self {
            start,
            stop,
            mut tags,
            states,
        } = self;
        tags.insert(stop, tag);
        Self {
            start,
            stop,
            tags,
            states,
        }
    }

    /// Number of states inside NFA
    pub fn size(&self) -> usize {
        self.states.len()
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
            tags: Default::default(),
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
            tags: Default::default(),
            states,
        }
    }

    /// Thompson's construction of NFAs chain one after another.
    /// For `a` and `b` regular expressions it is equivalent to `ab` expressions.
    pub fn sequence(nfas: impl IntoIterator<Item = Self>) -> Self {
        let (mut states, ends, tags) = Self::merge_states(nfas, 0);
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
            tags,
            states,
        }
    }

    /// Thompson's construction of NFA that matches any of the provided NFAs
    /// For `a` and `b` regular expressions it is equivalent to `(a|b)` expressions.
    pub fn choice(nfas: impl IntoIterator<Item = Self>) -> Self {
        let (mut states, ends, tags) = Self::merge_states(nfas, 2);
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
            tags,
            states,
        }
    }

    /// For `a` regular expression it is equivalent to `a+`
    pub fn some(self) -> Self {
        todo!()
    }

    /// For `a` regular expression it is equivalent to `a*`
    pub fn many(self) -> Self {
        // add offset of 2 to state ids
        let (mut states, ends, tags) = Self::merge_states(once(self), 2);
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
            tags,
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
        BTreeMap<NFAStateId, NFAState>,
        // starts/stops
        Vec<(NFAStateId, NFAStateId)>,
        // tags
        BTreeMap<NFAStateId, T>,
    ) {
        let mut states_out: BTreeMap<NFAStateId, NFAState> = BTreeMap::new();
        let mut ends_out: Vec<(NFAStateId, NFAStateId)> = Vec::new();
        let mut tags_out: BTreeMap<NFAStateId, T> = BTreeMap::new();

        for NFA {
            start,
            stop,
            states,
            tags,
        } in nfas
        {
            let start = NFAStateId(offset + start.0);
            let stop = NFAStateId(offset + stop.0);
            ends_out.push((start, stop));

            let mut max_id = 0;
            for (id, state) in states {
                max_id = std::cmp::max(max_id, id.0);
                let NFAState { edges, epsilons } = state;
                let id = NFAStateId(offset + id.0);
                let edges = edges
                    .into_iter()
                    .map(|(k, v)| (k, NFAStateId(offset + v.0)))
                    .collect();
                let epsilons = epsilons
                    .into_iter()
                    .map(|v| NFAStateId(offset + v.0))
                    .collect();
                states_out.insert(id, NFAState { edges, epsilons });
            }

            for (state, tag) in tags {
                tags_out.insert(NFAStateId(offset + state.0), tag);
            }

            offset += max_id + 1;
        }
        (states_out, ends_out, tags_out)
    }

    /// NFA to DFA using powerset construction
    pub fn compile(&self) -> DFA<T>
    where
        T: Clone + Ord,
    {
        // each DFA state is represented as epsilon closure of NFA states `Rc<BTreeSet<NFAStateId>>`
        let mut dfa_states: BTreeMap<NFAStateSet, DFAStateId> = BTreeMap::new();

        // constructed DFA
        let mut dfa_table: BTreeMap<DFAStateId, BTreeMap<Symbol, DFAStateId>> = BTreeMap::new();
        let mut dfa_stops: BTreeSet<DFAStateId> = BTreeSet::new();

        // initialize traversal queue with initial state
        let dfa_start = self.epsilon_closure(once(self.start));
        let dfa_start_id = DFAStateId(0);
        let mut dfa_queue = vec![(dfa_start_id, dfa_start.clone())]; // to be traversed DFA states
        dfa_states.insert(dfa_start, DFAStateId(0));

        while let Some((dfa_state_id, dfa_state)) = dfa_queue.pop() {
            // check if this DFA state contains final NFA state
            let is_final = dfa_state.contains(&self.stop);

            // find all unique symbols leading from the current DFA state
            let symbols: BTreeSet<Symbol> = dfa_state
                .iter()
                .flat_map(|nfa_state_id| self.states[nfa_state_id].edges.keys().copied())
                .collect();

            // resolve all edges of the current DFA state
            let mut dfa_edges: BTreeMap<Symbol, DFAStateId> = BTreeMap::new();
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
                        let id = DFAStateId(dfa_states.len());
                        dfa_states.insert(dfa_state_new.clone(), id);
                        dfa_queue.push((id, dfa_state_new));
                        id
                    }
                };
                // update edges of the current DFA state
                dfa_edges.insert(symbol, dfa_state_new_id);
            }

            // update DFA
            if is_final {
                dfa_stops.insert(dfa_state_id);
            }
            dfa_table.insert(dfa_state_id, dfa_edges);
        }

        // by construction all states are dense (meaning they start from 0 and do not have gaps)
        let lang_size = Symbol::MAX as usize + 1;
        let states = dfa_table
            .into_iter()
            .enumerate()
            .flat_map(|(index, (state, edges))| {
                assert_eq!(index, state.0);
                (0..lang_size).map(move |symbol| edges.get(&(symbol as Symbol)).copied())
            })
            .collect();

        let mut tags: Vec<BTreeSet<T>> = Vec::new();
        tags.resize_with(dfa_states.len(), Default::default);
        for (nfa_state_set, dfa_state_id) in dfa_states {
            let dfa_tag = &mut tags[dfa_state_id.0];
            for nfa_state_id in nfa_state_set.iter() {
                if let Some(tag) = self.tags.get(nfa_state_id) {
                    dfa_tag.insert(tag.clone());
                }
            }
        }

        DFA {
            lang_size,
            states,
            start: dfa_start_id,
            stops: dfa_stops,
            tags,
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
            tags: Default::default(),
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

impl<T> fmt::Debug for NFA<T> {
    /// Format NFA as a valid DOT graph
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\ndigraph NFA {{\n")?;
        write!(f, "  rankdir=\"LR\"\n")?;
        for (from, state) in self.states.iter() {
            if from == &self.stop {
                write!(f, "  {} [shape=doublecircle]\n", from.0)?;
            } else {
                write!(f, "  {} [shape=circle]\n", from.0)?;
            }
            for (symbol, to) in state.edges.iter() {
                write!(
                    f,
                    "  {} -> {} [label=\"{}\"]\n",
                    from.0,
                    to.0,
                    char::from(*symbol)
                )?;
            }
            for to in state.epsilons.iter() {
                write!(f, "  {} -> {} [color=red]\n", from.0, to.0)?;
            }
        }
        write!(f, "}}\n")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DFAStateId(usize);

pub struct DFA<T> {
    lang_size: usize,
    states: Vec<Option<DFAStateId>>,
    start: DFAStateId,
    stops: BTreeSet<DFAStateId>,
    tags: Vec<BTreeSet<T>>,
}

impl<T> DFA<T> {
    /// Nuber of states in DFA
    pub fn size(&self) -> usize {
        self.states.len() / self.lang_size
    }

    /// Get start state of the DFA
    pub fn start(&self) -> DFAStateId {
        self.start
    }

    /// Check if provided state is an accepting state
    pub fn is_accepting(&self, state: DFAStateId) -> bool {
        self.stops.contains(&state)
    }

    /// Get tags assocciated with the state
    pub fn tags(&self, state: DFAStateId) -> &BTreeSet<T> {
        &self.tags[state.0]
    }

    /// Transition from provided state given symbol
    pub fn transition(&self, state: DFAStateId, symbol: Symbol) -> Option<DFAStateId> {
        self.states[self.lang_size * state.0 + symbol as usize]
    }

    pub fn transition_many(
        &self,
        state: DFAStateId,
        symbols: impl IntoIterator<Item = Symbol>,
    ) -> Option<DFAStateId> {
        symbols
            .into_iter()
            .try_fold(state, |state, symbol| self.transition(state, symbol))
    }

    /// Check if DFA accepts given input
    pub fn matches(&self, symbols: impl IntoIterator<Item = Symbol>) -> bool {
        if let Some(state) = self.transition_many(self.start(), symbols) {
            self.is_accepting(state)
        } else {
            false
        }
    }
}

impl<T> fmt::Debug for DFA<T> {
    /// Format NFA as a valid DOT graph
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\ndigraph DFA {{\n")?;
        write!(f, "  rankdir=\"LR\"\n")?;
        for from in (0..self.size()).map(DFAStateId) {
            if self.stops.contains(&from) {
                write!(f, "  {} [shape=doublecircle]\n", from.0)?;
            } else {
                write!(f, "  {} [shape=circle]\n", from.0)?;
            }
            for symbol in 0..self.lang_size {
                let symbol = symbol as Symbol;
                match self.transition(from, symbol) {
                    None => continue,
                    Some(to) => {
                        write!(
                            f,
                            "  {} -> {} [label=\"{}\"]\n",
                            from.0,
                            to.0,
                            char::from(symbol)
                        )?;
                    }
                }
            }
        }
        write!(f, "}}\n")?;
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
        assert_eq!(dfa.tags(state), &once(1).collect::<BTreeSet<_>>());

        let state = dfa
            .transition_many(dfa.start(), "abd".bytes())
            .expect("should match");
        assert_eq!(dfa.tags(state), &once(2).collect::<BTreeSet<_>>());
    }

    #[test]
    fn test_nfa_from_str() {
        let nfa: NFA<()> = NFA::from("abc");
        assert_eq!(nfa.size(), 4);
    }
}
