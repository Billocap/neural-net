import * as math from "mathjs";

import Sketch from "../lib/Sketch";
import NeuralNet from "../lib/NeuralNet";
import Dataset from "../lib/Dataset";

class DrawTraining extends Sketch {
  private _width: number;
  private _height: number;
  private dataset: Dataset;
  private neuralNet: NeuralNet;

  constructor(
    width: number,
    height: number,
    dataset: Dataset,
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
    document.title = Math.round(this.frameRate()).toString();

    this.background(128);

    const training = this.neuralNet.train(this.dataset);

    let error = 0;

    for (const [point, r] of training) {
      this.strokeWeight(4);

      this.stroke(
        point.label[0] * 128,
        point.label[1] * 128,
        point.label[2] * 128
      );
      this.point(point.value[0] * 400, point.value[1] * 400);

      error += math.subtract(r, point.label).reduce((a, n) => a + n * n);

      this.strokeWeight(3);

      this.stroke(r[0] * 255, r[1] * 255, r[2] * 255);
      this.point(point.value[0] * 400, point.value[1] * 400);
    }

    error /= this.dataset.length;

    this.strokeWeight(0);
    this.fill("white");

    this.text(error, 10, 20);

    this.dataset.shuffle();

    // if (error < 0.009) {
    //   console.log(this.neuralNet.layers);

    //   this.noLoop();
    // }
  }
}

export default DrawTraining;
