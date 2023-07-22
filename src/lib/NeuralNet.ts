import * as math from "mathjs";

import Layer from "./Layer";

class NeuralNet {
  public layers: Layer[];

  constructor(...s: number[]) {
    this.layers = [];

    for (let l = 0; l < s.length - 1; l++) {
      const layer = new Layer(s[l], s[l + 1]);

      layer.rate = 0.1;

      this.layers.push(layer);
    }
  }

  functions(fun: iActivation, prime: iActivation) {
    for (const layer of this.layers) {
      layer.functions(fun, prime);
    }
  }

  *ff(inputs: iVector): Generator<iVector, iGradient[], iVector> {
    const ffs = [this.layers[0].ff(inputs)];

    for (let l = 0; l < this.layers.length - 1; l++) {
      const pR = ffs[l].next().value as iVector;

      ffs.push(this.layers[l + 1].ff(pR));
    }

    let dC = yield ffs[this.layers.length - 1].next().value as iVector;

    const grad = [] as iGradient[];

    for (let l = ffs.length - 1; l >= 0; l--) {
      const [gW, gB] = ffs[l].next(dC).value as iGradient;

      dC = math.multiply(gB, math.transpose(this.layers[l].weights));

      grad.push([gW, gB]);
    }

    return grad.reverse();
  }

  *train(): Generator<void, void, iGradient[] | undefined> {
    const ts = this.layers.map((l) => l.train());

    for (const t of ts) t.next();

    while (true) {
      const grad = yield;

      if (grad) {
        grad.forEach((g, i) => ts[i].next(g));
      } else {
        for (const t of ts) t.next();

        break;
      }
    }
  }
}

export default NeuralNet;
