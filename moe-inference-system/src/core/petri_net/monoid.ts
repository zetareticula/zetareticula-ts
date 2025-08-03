import { PetriNetState, PetriNetTransition } from './types';

export class PetriNetMonoid {
  private transitions: PetriNetTransition[] = [];

  addTransition(from: PetriNetState, to: PetriNetState): void {
    this.transitions.push({ from, to });
  }

  getWeight(state: PetriNetState): number {
    return state.weight;
  }
}