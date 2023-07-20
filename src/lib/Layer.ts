class Layer {
  public weights: number[][];
  public biases: number[];
  public rate: number;

  constructor(inputs: number, outputs: number) {
    this.weights = [];
    this.biases = [];

    for (let y = 0; y < outputs; y++) {
      const row = [];

      for (let x = 0; x < inputs; x++) row.push(1 ?? Math.random());

      this.weights.push(row);

      this.biases.push(1 ?? Math.random());
    }

    this.rate = 1;
  }

  // active(x: number) {
  //   return (2 * Math.atan(x)) / Math.PI;
  // }

  // pActive(x: number) {
  //   return 2 / (Math.PI * (1 + x * x));
  // }

  active(x: number) {
    return 1 / (1 + Math.exp(-x));
  }

  pActive(x: number) {
    return this.active(x) * (1 - this.active(x));
  }

  *feedForward(...inputs: number[]): Generator<number[], iGradient, number[]> {
    const zs = this.weights.map((row, y) => {
      return (
        row.reduce((a, w, x) => {
          return a + w * inputs[x];
        }, 0) + this.biases[y]
      );
    });

    const as = zs.map((x) => this.active(x));

    const dC = yield as;

    const dA = zs.map((x) => this.pActive(x));

    const gradW = dC.map((C, x) => inputs.map((i) => i * dA[x] * C));

    const gradB = dC.map((C, x) => C * dA[x]);

    return {
      weights: gradW,
      biases: gradB
    };
  }

  train({ weights, biases }: iGradient) {
    this.weights = weights.map((row, y) =>
      row.map((w, x) => this.weights[y][x] - w * this.rate)
    );

    this.biases = biases.map((b, i) => this.biases[i] - b * this.rate);
  }
}

export default Layer;
