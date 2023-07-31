import * as math from "mathjs";

class Layer implements Model.Layer {
  public weights: Matrix;
  public biases: Vector;
  public rate: number;
  public size: [number, number];

  private active: Model.Activation;
  private pActive: Model.Activation;

  constructor(m: number, n: number) {
    this.size = [m, n];

    this.weights = math.map(math.zeros([m, n]), () => math.random()) as Matrix;
    this.biases = math.map(math.zeros([n]), () => math.random()) as Vector;

    this.rate = 0.1;

    this.active = (n) => n;
    this.pActive = (n) => n;
  }

  functions(fun: Model.Activation, prime: Model.Activation) {
    this.active = fun;
    this.pActive = prime;
  }

  *ff(inputs: Vector): Generator<Vector, Model.Gradient, Vector> {
    const { weights, biases, active, pActive } = this;

    const zs = math.add(math.multiply(inputs, weights), biases);

    const dC = yield math.map(zs, active);

    const dA = math.map(zs, pActive);

    const gradB = dA.map((a, i) => a * dC[i]);

    const gradW = inputs.map((i) => gradB.map((g) => i * g));

    return [gradW, gradB];
  }

  *train(): Generator<void, void, Model.Gradient | undefined> {
    const [m, n] = this.size;

    const avg = [math.zeros([m, n]), math.zeros([n])] as Model.Gradient;

    let s = 0;

    while (true) {
      const r = yield;

      if (r) {
        const [gradW, gradB] = r;

        avg[0] = math.add(avg[0], gradW) as Matrix;
        avg[1] = math.add(avg[1], gradB) as Vector;
      } else {
        const { weights, biases, rate } = this;

        avg[0] = math.multiply(avg[0], 1 / s) as Matrix;
        avg[1] = math.multiply(avg[1], 1 / s) as Vector;

        const [gW, gB] = avg;

        this.weights = <Matrix>math.subtract(weights, math.multiply(gW, rate));
        this.biases = <Vector>math.subtract(biases, math.multiply(gB, rate));

        break;
      }

      s++;
    }
  }
}

export default Layer;
