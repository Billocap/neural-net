import * as math from "mathjs";

import Layer from "../lib/Layer";
import Sketch from "../lib/Sketch";

class DrawTraining extends Sketch {
  private _width: number;
  private _height: number;
  private dataset: iData[];
  private layers: Layer[];

  constructor(
    width: number,
    height: number,
    dataset: iData[],
    layers: Layer[]
  ) {
    super();

    this._width = width;
    this._height = height;
    this.dataset = dataset;
    this.layers = layers;
  }

  setup() {
    this.createCanvas(this._width, this._height);
  }

  draw() {
    this.background(128);

    this.strokeWeight(5);

    for (const point of this.dataset) {
      this.stroke(point.label[0] * 128, point.label[1] * 128, 0);
      this.point(point.value[0] * 400, point.value[1] * 400);
    }

    this.strokeWeight(3);

    const t0 = this.layers[0].train();
    const t1 = this.layers[1].train();

    t0.next();
    t1.next();

    for (const point of this.dataset) {
      const ff0 = this.layers[0].ff(point.value);

      const r0 = ff0.next().value as iVector;

      const ff1 = this.layers[1].ff(r0);

      const r = ff1.next().value as iVector;

      this.stroke(r[0] * 255, r[1] * 255, 128);
      this.point(point.value[0] * 400, point.value[1] * 400);

      let dC = math.multiply(2, math.subtract(r, point.label)) as iVector;

      const [gW, gB] = ff1.next(dC).value as iGradient;

      t1.next([gW, gB]);

      dC = math.multiply(gB, math.transpose(this.layers[1].weights));

      const [g0W, g0B] = ff0.next(dC).value as iGradient;

      t0.next([g0W, g0B]);
    }

    t0.next();
    t1.next();
  }
}

export default DrawTraining;
