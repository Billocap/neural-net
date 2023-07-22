import DrawTraining from "./sketches/DrawTraining";
import Layer from "./lib/Layer";
import Dataset from "./lib/Dataset";

import "./style.css";

const generate = () => [Math.random(), Math.random()];

const m = Math.random() * 2 - 1;

const n = Math.random();

const lineClassifier = ([x, y]: iVector) => {
  return m * x + n > y ? [1, 0] : [0, 1];
};

const circleClassifier = ([x, y]: iVector) => {
  return x * x + y * y > 0.5 * 0.5 ? [1, 0] : [0, 1];
};

const dataset = new Dataset(500);

dataset.generate(generate, circleClassifier);

const arctan = (x: number) => (2 * Math.atan(x)) / Math.PI;
const arctanPrime = (x: number) => 2 / (Math.PI * (1 + x * x));

const sigma = (x: number) => 1 / (1 + Math.exp(-x));
const sigmaPrime = (x: number) => sigma(x) * (1 - sigma(x));

const ls = [new Layer(2, 3), new Layer(3, 2)];

ls[0].functions(sigma, sigmaPrime);
ls[1].functions(sigma, sigmaPrime);

new DrawTraining(400, 400, dataset, ls);
