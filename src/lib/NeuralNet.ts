import * as math from "mathjs";

class NeuralNet {
  public weights: iMatrix;
  public biases: iVector;
  public rate: number;

  constructor(...s: number[]) {
    this.weights = math.map(math.zeros(s), () => math.random()) as iMatrix;
    this.biases = math.map(math.zeros([s[1]]), () => math.random()) as iVector;

    this.rate = 0.1;
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

  *ff(...inputs: iVector): Generator<iVector, [iMatrix, iVector], iVector> {
    const wSum = math.multiply(inputs, this.weights);

    const zs = math.add(wSum, this.biases);

    const dC = yield math.map(zs, (x) => this.active(x));

    const dA = math.map(zs, (x) => this.pActive(x));

    const gradB = dA.map((a, i) => a * dC[i]);

    const gradW = inputs.map((i) => gradB.map((g) => i * g));

    return [gradW, gradB];
  }

  train([gradW, gradB]: [iMatrix, iVector]) {
    const { weights, biases, rate } = this;

    this.weights = <iMatrix>math.subtract(weights, math.multiply(gradW, rate));
    this.biases = <iVector>math.subtract(biases, math.multiply(gradB, rate));
  }
}

export default NeuralNet;
