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

    this.strokeWeight(4);

    for (const point of this.dataset) {
      this.stroke(point.label > 0 ? "red" : "green");
      this.point(point.x, point.y);
    }

    this.strokeWeight(3);

    let avg = [0, 0, 0];

    for (const point of this.dataset) {
      const ff = this.neuralNet.feedforward(point.x / 400, point.y / 400);

      const c = ff.next().value as number;

      this.stroke((c * 0.5 + 0.5) * 255);
      this.point(point.x, point.y);

      const grad = ff.next(point.label).value as number[];

      avg = grad.map((n, i) => n + avg[i]);
    }

    avg = avg.map((n) => n / avg.length);

    this.neuralNet.train(...avg);
  }
}

export default DrawTraining;
