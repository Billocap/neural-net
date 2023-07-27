import p5 from "p5";

import Sketch from "../lib/Sketch";
import NeuralNet from "../lib/NeuralNet";

import vertexShader from "../assets/p5.vert?raw";
import fragmentShader from "../assets/nn.frag?raw";

let shader: p5.Shader;

class DrawDomain extends Sketch {
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
    this.createCanvas(this._width, this._height, this.WEBGL);

    this.strokeWeight(0);

    shader = this.createShader(vertexShader, fragmentShader);
  }

  draw() {
    this.shader(shader);

    const uniforms = {
      w_sizes: [] as number[],
      weights: [] as number[],
      b_sizes: [] as number[],
      biases: [] as number[]
    };

    for (const { weights, biases, size } of this.neuralNet.layers) {
      uniforms.w_sizes.push(...size);
      for (const row of weights) uniforms.weights.push(...row);

      uniforms.b_sizes.push(size[1]);
      uniforms.biases.push(...biases);
    }

    shader.setUniform("resolution", [400, 400]);

    shader.setUniform("w_sizes", uniforms.w_sizes);
    shader.setUniform("weights", uniforms.weights);

    shader.setUniform("b_sizes", uniforms.b_sizes);
    shader.setUniform("biases", uniforms.biases);

    this.rect(0, 0, this.width, this.height);
  }
}

export default DrawDomain;
