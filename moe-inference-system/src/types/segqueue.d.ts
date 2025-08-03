declare module 'segqueue' {
  export class SegQueue<T = any> {
    constructor();
    push(item: T): void;
    shift(): T | undefined;
    isEmpty(): boolean;
    clear(): void;
    get length(): number;
  }
}
