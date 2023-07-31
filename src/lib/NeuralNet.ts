import * as math from "mathjs";

import Layer from "./Layer";

class NeuralNet implements Model.Model {
  public layers: Layer[];

  constructor(...s: number[]) {
    this.layers = [];

    for (let l = 0; l < s.length - 1; l++) {
      const layer = new Layer(s[l], s[l + 1]);

      layer.rate = 1;

      this.layers.push(layer);
    }
  }

  functions(fun: Model.Activation, prime: Model.Activation) {
    for (const layer of this.layers) {
      layer.functions(fun, prime);
    }
  }

  *ff(inputs: Vector): Generator<Vector, Model.Gradient[], Vector> {
    const ffs = [this.layers[0].ff(inputs)];

    for (let l = 0; l < this.layers.length - 1; l++) {
      const pR = ffs[l].next().value as Vector;

      ffs.push(this.layers[l + 1].ff(pR));
    }

    let dC = yield ffs[this.layers.length - 1].next().value as Vector;

    const grad = [] as Model.Gradient[];

    for (let l = ffs.length - 1; l >= 0; l--) {
      const [gW, gB] = ffs[l].next(dC).value as Model.Gradient;

      dC = math.multiply(gB, math.transpose(this.layers[l].weights));

      grad.push([gW, gB]);
    }

    return grad.reverse();
  }

  *train(dataset: Data.Set): Generator<[Data.Data, Vector], void, unknown> {
    const ts = this.layers.map((l) => l.train());

    for (const t of ts) t.next();

    for (const data of dataset) {
      const ff = this.ff(data.value);

      const r = ff.next().value as Vector;

      yield [data, r];

      let dC = math.multiply(2, math.subtract(r, data.label)) as Vector;

      const grad = ff.next(dC).value as Model.Gradient[];

      grad.forEach((g, i) => ts[i].next(g));
    }

    for (const t of ts) t.next();
  }
}

export default NeuralNet;
