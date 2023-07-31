import NeuralNet from "../lib/NeuralNet";
import Sketch from "../lib/Sketch";

class DrawMouseClass extends Sketch {
  private _width: number;
  private _height: number;
  private neuralNet: NeuralNet;

  constructor(width: number, height: number, neuralNet: NeuralNet) {
    super();

    this._width = width;
    this._height = height;
    this.neuralNet = neuralNet;
  }

  setup() {
    this.createCanvas(this._width, this._height);
  }

  draw() {
    this.background(128);

    const ff = this.neuralNet.ff([this.mouseX / 400, this.mouseY / 400]);

    const [r, g, b] = ff.next().value as Vector;

    this.strokeWeight(5);

    this.stroke(r * 255, g * 255, b * 255);
    this.point(this.mouseX, this.mouseY);

    this.strokeWeight(0);

    this.text([r, g, b].join("\n"), 10, 20);
  }
}
export default DrawMouseClass;
