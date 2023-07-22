import DrawTraining from "./sketches/DrawTraining";
import NeuralNet from "./lib/NeuralNet";
import Dataset from "./lib/Dataset";

import "./style.css";

const generate = () => [Math.random() * 400, Math.random() * 400];

const m = Math.random() * 2 - 1;

const n = Math.random() * 400;

const lineClassifier = ([x, y]: iVector) => {
  return m * x + n > y ? [1, 0] : [0, 1];
};

const circleClassifier = ([x, y]: iVector) => {
  return x * x + y * y > 200 * 200 ? [1, 0] : [0, 1];
};

const dataset = new Dataset(500);

dataset.generate(generate, circleClassifier);

const arctan = (x: number) => (2 * Math.atan(x)) / Math.PI;
const arctanPrime = (x: number) => 2 / (Math.PI * (1 + x * x));

const sigma = (x: number) => 1 / (1 + Math.exp(-x));
const sigmaPrime = (x: number) => sigma(x) * (1 - sigma(x));

const nn = new NeuralNet(2, 2);

nn.functions(sigma, sigmaPrime);

new DrawTraining(400, 400, dataset, nn);
