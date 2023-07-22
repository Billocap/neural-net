import DrawTraining from "./sketches/DrawTraining";
import Layer from "./lib/Layer";
import Dataset from "./lib/Dataset";

import "./style.css";
import NeuralNet from "./lib/NeuralNet";

const generate = () => [Math.random(), Math.random()];

const m = Math.random() * 2 - 1;

const n = Math.random();

const lineClassifier = ([x, y]: iVector) => {
  return m * x + n > y ? [1, 0] : [0, 1];
};

const circleClassifier = ([x, y]: iVector) => {
  return (x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.4 ** 2 ? [1, 0] : [0, 1];
};

const dataset = new Dataset(500);

dataset.generate(generate, circleClassifier);

const arctan = (x: number) => (2 * Math.atan(x)) / Math.PI;
const arctanPrime = (x: number) => 2 / (Math.PI * (1 + x * x));

const sigma = (x: number) => 1 / (1 + Math.exp(-x));
const sigmaPrime = (x: number) => sigma(x) * (1 - sigma(x));

const nn = new NeuralNet(2, 3, 3, 2);

nn.functions(sigma, sigmaPrime);

new DrawTraining(400, 400, dataset, nn);
