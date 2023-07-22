import * as math from "mathjs";

class NeuralNet {
  public weights: number[][];
  public biases: number[];
  public rate: number;

  constructor(...s: number[]) {
    this.weights = <number[][]>math.map(math.zeros(s), () => 1);
    this.biases = <number[]>math.map(math.zeros([s[1]]), () => 1);

    this.rate = 1;
  }

  active(x: number) {
    return (2 * Math.atan(x)) / Math.PI;
  }

  pActive(x: number) {
    return 2 / (Math.PI * (1 + x * x));
  }

  *ff(
    ...inputs: number[]
  ): Generator<number[], [number[][], number[]], number[]> {
    const wSum = math.multiply(inputs, this.weights);

    const zs = math.add(wSum, this.biases);

    const dC = yield math.map(zs, (x) => this.active(x));

    const dA = math.map(zs, (x) => this.pActive(x));

    const gradB = dA.map((a, i) => a * dC[i]);

    const gradW = inputs.map((i) => gradB.map((g) => i * g));

    return [gradW, gradB];
  }

  train(gradW: number[][], gradB: number[]) {
    const { weights, biases, rate } = this;

    this.weights = <number[][]>(
      math.subtract(weights, math.multiply(gradW, rate))
    );
    this.biases = <number[]>math.subtract(biases, math.multiply(gradB, rate));
  }
}

export default NeuralNet;
