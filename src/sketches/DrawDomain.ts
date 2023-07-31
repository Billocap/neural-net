import * as THREE from "three";
import * as math from "mathjs";

import Sketch from "@/lib/Sketch";

import vertexShader from "@/assets/p5.vert?raw";
import fragmentShader from "@/assets/nn.frag?raw";

class DrawDomain {
  private _width: number;
  private _height: number;
  private model: Model.Model;

  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.Renderer;
  private mesh: THREE.Mesh;
  private material: THREE.ShaderMaterial;

  constructor(width: number, height: number, model: Model.Model) {
    this._width = width;
    this._height = height;
    this.model = model;

    this.scene = new THREE.Scene();

    this.camera = new THREE.OrthographicCamera(
      -width / 2,
      width / 2,
      -height / 2,
      height / 2,
      1,
      1000
    );
    this.camera.position.z = 5;

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setSize(width, height);
    document.body.appendChild(this.renderer.domElement);

    const geometry = new THREE.PlaneGeometry(width, height);
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        resolution: {
          value: new THREE.Vector2(width, height)
        },
        sizes: {
          value: []
        },
        weights: {
          value: []
        },
        biases: {
          value: []
        }
      },
      vertexShader,
      fragmentShader
    });
    this.mesh = new THREE.Mesh(geometry, this.material);

    this.scene.add(this.mesh);
  }

  draw() {
    const uniforms = {
      resolution: {
        value: new THREE.Vector2(this._width, this._height)
      },
      sizes: [] as number[],
      weights: [] as number[],
      biases: [] as number[]
    };

    uniforms.sizes.push(this.model.layers[0].size[0]);

    for (const { weights, biases, size } of this.model.layers) {
      const ws = math.transpose(weights);

      for (const row of ws) uniforms.weights.push(...row);

      uniforms.biases.push(...biases);

      uniforms.sizes.push(size[1]);
    }

    this.material.uniforms = {
      resolution: {
        value: new THREE.Vector2(this._width, this._height)
      },
      sizes: {
        value: new Uint8Array(uniforms.sizes)
      },
      weights: {
        value: new Float64Array(uniforms.weights)
      },
      biases: {
        value: new Float64Array(uniforms.biases)
      }
    };

    this.renderer.render(this.scene, this.camera);

    requestAnimationFrame(() => this.draw());
  }
}

export default DrawDomain;
