export interface PetriNetState {
    place: 'store_init' | 'store_hbm' | 'store_complete';
    weight: number;
  }
  
  export interface PetriNetTransition {
    from: PetriNetState;
    to: PetriNetState;
  }