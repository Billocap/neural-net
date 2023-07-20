import Layer from "../lib/Layer";
import Sketch from "../lib/Sketch";

class DrawTraining extends Sketch {
  private _width: number;
  private _height: number;
  private dataset: Point[];
  private neuralNet: Layer;

  constructor(
    width: number,
    height: number,
    dataset: Point[],
    neuralNet: Layer
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
      this.stroke(point.label[0] * 255, point.label[1] * 255, 0);
      this.point(point.x, point.y);
    }

    this.strokeWeight(3);

    const avg: iGradient = {
      weights: [
        [0, 0],
        [0, 0]
      ],
      biases: [0, 0]
    };

    for (const point of this.dataset) {
      const ff = this.neuralNet.feedForward(point.x / 400, point.y / 400);

      const [c, b] = ff.next().value as number[];

      this.stroke(c * 255, b * 255, 128);
      this.point(point.x, point.y);

      const grad = ff.next([2 * (c - point.label[0]), 2 * (b - point.label[1])])
        .value as iGradient;

      avg.weights = grad.weights.map((row, y) =>
        row.map((n, x) => avg.weights[y][x] + n)
      );

      avg.biases = grad.biases.map((b, i) => avg.biases[i] + b);
    }

    avg.weights = avg.weights.map((row) =>
      row.map((w) => w / this.dataset.length)
    );

    avg.biases = avg.biases.map((b) => b / this.dataset.length);

    this.neuralNet.train(avg);
  }
}

export default DrawTraining;
