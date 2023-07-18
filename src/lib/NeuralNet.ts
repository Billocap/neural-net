class NeuralNet {
  public ws: number[];
  public b: number;
  public rate: number;

  constructor() {
    this.ws = [Math.random() * 2 - 1, Math.random() * 2 - 1];

    this.b = Math.random();

    this.rate = 0.1;
  }

  active(x: number) {
    return (2 * Math.atan(x)) / Math.PI;
  }

  pActive(x: number) {
    return 2 / (Math.PI * (1 + x * x));
  }

  *feedforward(...inputs: number[]): Generator<number, number[], number> {
    const z = this.ws[0] * inputs[0] + this.ws[1] * inputs[1] + this.b;

    const a = this.active(z);

    const y = yield a;

    const dC = 2 * (a - y);

    const dA = this.pActive(z);

    return [dC * dA * inputs[0], dC * dA * inputs[1], dC * dA];
  }

  train(...grad: number[]) {
    const [w1, w2, b] = grad;

    this.ws[0] -= w1 * this.rate;
    this.ws[1] -= w2 * this.rate;
    this.b -= b * this.rate;
  }
}

export default NeuralNet;
