import p5 from "p5";
import * as math from "mathjs";

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
      sizes: [] as number[],
      weights: [] as number[],
      biases: [] as number[]
    };

    uniforms.sizes.push(this.neuralNet.layers[0].size[0]);

    for (const { weights, biases, size } of this.neuralNet.layers) {
      const ws = math.transpose(weights);

      for (const row of ws) uniforms.weights.push(...row);

      uniforms.biases.push(...biases);

      uniforms.sizes.push(size[1]);
    }

    shader.setUniform("resolution", [400, 400]);

    shader.setUniform("sizes", uniforms.sizes);
    shader.setUniform("weights", uniforms.weights);
    shader.setUniform("biases", uniforms.biases);

    shader.setUniform("result", []);

    this.rect(0, 0, this.width, this.height);
  }
}

export default DrawDomain;
