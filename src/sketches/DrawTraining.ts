import * as math from "mathjs";

import NeuralNet from "../lib/NeuralNet";
import Sketch from "../lib/Sketch";

class DrawTraining extends Sketch {
  private _width: number;
  private _height: number;
  private dataset: Point[];
  private neuralNet: NeuralNet;

  constructor(
    width: number,
    height: number,
    dataset: Point[],
    neuralNet: NeuralNet
  ) {
    super();

    this._width = width;
    this._height = height;
    this.dataset = dataset;
    this.neuralNet = neuralNet;
  }

  setup() {
    this.createCanvas(this._width, this._height);
  }

  draw() {
    this.background(128);

    this.strokeWeight(5);

    for (const point of this.dataset) {
      this.stroke(point.label[0] * 128, point.label[1] * 128, 0);
      this.point(point.x, point.y);
    }

    this.strokeWeight(3);

    const t = this.neuralNet.train();

    t.next();

    for (const point of this.dataset) {
      const ff = this.neuralNet.ff(point.x / 400, point.y / 400);

      const r = ff.next().value as iVector;

      this.stroke(r[0] * 255, r[1] * 255, 128);
      this.point(point.x, point.y);

      const dC = math.multiply(2, math.subtract(r, point.label)) as iVector;

      const [gW, gB] = ff.next(dC).value as [iMatrix, iVector];

      t.next([gW, gB]);
    }

    const s = 1 / this.dataset.length;

    t.next(s);
  }
}

export default DrawTraining;
