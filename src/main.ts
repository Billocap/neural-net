import DrawTraining from "./sketches/DrawTraining";
import DrawDomain from "./sketches/DrawDomain";

import Dataset from "./lib/Dataset";
import StateManager from "./lib/StateManager";
import NeuralNet from "./lib/NeuralNet";

import "./styles/globals.css";

const generateCircle = () => {
  return [(2 * Math.random() - 1) ** 2, (2 * Math.random() - 1) ** 2];
};

const generate = () => {
  return [Math.random(), Math.random()];
};

const m = 0.5 ?? Math.random() * 2 - 1;

const n = 0.5 ?? Math.random();

const lineClassifier = ([x, y]: iVector) => {
  return m * x < y ? (m * x + n > y ? [1, 0, 0] : [0, 1, 0]) : [0, 0, 1];
};

const circleClassifier = ([x, y]: iVector) => {
  return x ** 2 + y ** 2 > 0.5 ** 2 ? [1, 0, 0] : [0, 1, 0];
};

const dataset = new Dataset(2500);

dataset.generate(generate, lineClassifier);

const arctan = (x: number) => (2 * Math.atan(x)) / Math.PI;
const arctanPrime = (x: number) => 2 / (Math.PI * (1 + x * x));

const sigma = (x: number) => 1 / (1 + Math.exp(-x));
const sigmaPrime = (x: number) => sigma(x) * (1 - sigma(x));

const nn = new NeuralNet(2, 3, 3);

nn.functions(sigma, sigmaPrime);

// new DrawMouseClass(400, 400, nn);

new DrawTraining(400, 400, dataset, nn);

new DrawDomain(400, 400, nn).draw();

// setInterval(() => {
//   StateManager.save("autosave", nn);
// }, 1000);

document.querySelector("#save-state")?.addEventListener("click", () => {
  StateManager.save("model", nn);
});

document.querySelector("#export-state")?.addEventListener("click", () => {
  StateManager.export(nn);
});

document.querySelector("#load-state")?.addEventListener("click", () => {
  StateManager.load("model", nn);
});

document.querySelector("#import-state")?.addEventListener("input", (e) => {
  const { files } = e.target as HTMLInputElement;

  if (files) {
    StateManager.import(files[0], nn);
  }
});
