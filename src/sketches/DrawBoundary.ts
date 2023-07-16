import p5 from "p5";

import Sketch from "../lib/Sketch";

const sliders: { [index: string]: p5.Element } = {};

let shader: p5.Shader;

class DrawBoundary extends Sketch {
  preload() {
    shader = this.loadShader("shaders/main.vert", "shaders/nn.frag");
  }

  setup() {
    this.createCanvas(256, 256, this.WEBGL);

    sliders.w1 = this.createSlider(-1, 1, 0, 0.01);
    sliders.w2 = this.createSlider(-1, 1, 0, 0.01);
    sliders.b = this.createSlider(-10, 10, 0, 0.01);

    this.strokeWeight(0);
  }

  draw() {
    this.shader(shader);

    const w1 = sliders.w1.value() as number;
    const w2 = sliders.w2.value() as number;
    const b = sliders.b.value() as number;

    shader.setUniform("u_values", [w1, w2, b]);

    this.rect(0, 0, this.width, this.height);
  }
}

export default DrawBoundary;
