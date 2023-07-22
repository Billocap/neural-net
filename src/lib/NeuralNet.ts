import * as math from "mathjs";

class NeuralNet {
  public weights: iMatrix;
  public biases: iVector;
  public rate: number;

  private size: number[];
  private active: iActivation;
  private pActive: iActivation;

  constructor(...s: number[]) {
    this.size = s;

    this.weights = math.map(math.zeros(s), () => math.random()) as iMatrix;
    this.biases = math.map(math.zeros([s[1]]), () => math.random()) as iVector;

    this.rate = 0.1;

    this.active = (n) => n;
    this.pActive = (n) => n;
  }

  functions(fun: iActivation, prime: iActivation) {
    this.active = fun;
    this.pActive = prime;
  }

  *ff(inputs: iVector): Generator<iVector, iGradient, iVector> {
    const { weights, biases, active, pActive } = this;

    const zs = math.add(math.multiply(inputs, weights), biases);

    const dC = yield math.map(zs, active);

    const dA = math.map(zs, pActive);

    const gradB = dA.map((a, i) => a * dC[i]);

    const gradW = inputs.map((i) => gradB.map((g) => i * g));

    return [gradW, gradB];
  }

  *train(): Generator<void, void, iGradient | undefined> {
    const [m, n] = this.size;

    const avg = [math.zeros([m, n]), math.zeros([n])] as iGradient;

    let s = 0;

    while (true) {
      const r = yield;

      if (r) {
        const [gradW, gradB] = r;

        avg[0] = math.add(avg[0], gradW) as iMatrix;
        avg[1] = math.add(avg[1], gradB) as iVector;
      } else {
        const { weights, biases, rate } = this;

        avg[0] = math.multiply(avg[0], 1 / s) as iMatrix;
        avg[1] = math.multiply(avg[1], 1 / s) as iVector;

        const [gW, gB] = avg;

        this.weights = <iMatrix>math.subtract(weights, math.multiply(gW, rate));
        this.biases = <iVector>math.subtract(biases, math.multiply(gB, rate));

        break;
      }

      s++;
    }
  }
}

export default NeuralNet;
