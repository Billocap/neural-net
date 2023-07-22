import * as math from "mathjs";

class NeuralNet {
  public weights: iMatrix;
  public biases: iVector;
  public rate: number;

  private size: number[];

  constructor(...s: number[]) {
    this.size = s;

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

  *ff(...inputs: iVector): Generator<iVector, iGradient, iVector> {
    const wSum = math.multiply(inputs, this.weights);

    const zs = math.add(wSum, this.biases);

    const dC = yield math.map(zs, (x) => this.active(x));

    const dA = math.map(zs, (x) => this.pActive(x));

    const gradB = dA.map((a, i) => a * dC[i]);

    const gradW = inputs.map((i) => gradB.map((g) => i * g));

    return [gradW, gradB];
  }

  *train(): Generator<void, void, iGradient | number> {
    const [m, n] = this.size;

    const avg = [math.zeros([m, n]), math.zeros([n])] as iGradient;

    while (true) {
      const r = yield;

      if (Array.isArray(r)) {
        const [gradW, gradB] = r;

        avg[0] = math.add(avg[0], gradW) as iMatrix;
        avg[1] = math.add(avg[1], gradB) as iVector;
      } else {
        const { weights, biases, rate } = this;

        avg[0] = math.multiply(avg[0], r) as iMatrix;
        avg[1] = math.multiply(avg[1], r) as iVector;

        const [gW, gB] = avg;

        this.weights = <iMatrix>math.subtract(weights, math.multiply(gW, rate));
        this.biases = <iVector>math.subtract(biases, math.multiply(gB, rate));

        break;
      }
    }
  }
}

export default NeuralNet;
