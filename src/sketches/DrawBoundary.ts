import p5 from "p5";

import Sketch from "../lib/Sketch";
import NeuralNet from "../lib/NeuralNet";

let shader: p5.Shader;

class DrawBoundary extends Sketch {
  private _width: number;
  private _height: number;
  private neuralNet: NeuralNet;

  constructor(width: number, height: number, neuralNet: NeuralNet) {
    super();

    this._width = width;
    this._height = height;
    this.neuralNet = neuralNet;
  }

  preload() {
    shader = this.loadShader("shaders/main.vert", "shaders/nn.frag");
  }

  setup() {
    this.createCanvas(this._width, this._height, this.WEBGL);

    this.strokeWeight(0);
  }

  draw() {
    this.shader(shader);

    shader.setUniform("values", [
      this.neuralNet.ws[0],
      this.neuralNet.ws[1],
      this.neuralNet.b
    ]);

    shader.setUniform("resolution", [400, 400]);

    this.rect(0, 0, this.width, this.height);
  }
}

export default DrawBoundary;
