use std::{
    collections::{BTreeSet, HashMap, HashSet},
    iter::once,
    rc::Rc,
};

type Symbol = u8;

trait Language {
    fn uid(&self) -> usize;
    fn max(&self) -> usize;
}

impl Language for Symbol {
    fn uid(&self) -> usize {
        *self as usize
    }

    fn max(&self) -> usize {
        Self::max_value() as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NFAStateId(usize);

type NFAStateSet = Rc<BTreeSet<NFAStateId>>;

struct NFAState {
    edges: HashMap<Symbol, NFAStateId>,
    epsilons: BTreeSet<NFAStateId>,
}

pub struct NFA {
    start: NFAStateId,
    stop: NFAStateId,
    states: HashMap<NFAStateId, NFAState>,
}

impl NFA {
    pub fn sequence(nfas: impl IntoIterator<Item = Self>) -> NFA {
        unimplemented!()
    }

    pub fn choice(nfas: impl IntoIterator<Item = Self>) -> NFA {
        unimplemented!()
    }

    pub fn some(self) -> NFA {
        unimplemented!()
    }

    pub fn many(self) -> NFA {
        unimplemented!()
    }

    /// Merge multiple NFAs
    ///
    /// - recalulate ids so ids from differnt NFAs would be different
    /// - merges states into a single mapping
    /// - merges stops and starts preserving order
    fn merge_state_ids(
        nfas: impl IntoIterator<Item = Self>,
    ) -> (
        // states
        HashMap<NFAStateId, NFAState>,
        // starts
        Vec<NFAStateId>,
        // stops
        Vec<NFAStateId>,
    ) {
        let mut states_out: HashMap<NFAStateId, NFAState> = HashMap::new();
        let mut starts_out: Vec<NFAStateId> = Vec::new();
        let mut stops_out: Vec<NFAStateId> = Vec::new();

        let mut offset = 0;
        for NFA {
            start,
            stop,
            states,
        } in nfas
        {
            starts_out.push(NFAStateId(offset + start.0));
            stops_out.push(NFAStateId(offset + stop.0));
            for (id, state) in states {
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

                /// TODO: UPDATE OFFSET!
                unimplemented!()
            }
        }
        (states_out, starts_out, stops_out)
    }

    /// NFA to DFA using powerset construction
    pub fn compile(&self) -> DFA {
        // each DFA state is represented as epsilon closure of NFA states `Rc<BTreeSet<NFAStateId>>`
        let mut dfa_states: HashMap<NFAStateSet, DFAStateId> = HashMap::new();

        // constructed DFA
        let mut dfa_table: HashMap<DFAStateId, HashMap<Symbol, DFAStateId>> = HashMap::new();
        let mut dfa_finals: HashSet<DFAStateId> = HashSet::new();

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
            let mut dfa_edges: HashMap<Symbol, DFAStateId> = HashMap::new();
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
            dfa_finals.insert(dfa_state_id);
            dfa_table.insert(dfa_state_id, dfa_edges);
        }

        DFA {}
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DFAStateId(usize);

struct DFA {}

#[cfg(test)]
mod tests {}
