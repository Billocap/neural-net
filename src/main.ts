import Layer from "./lib/Layer";
import DrawTraining from "./sketches/DrawTraining";

import "./style.css";

const m = Math.random() * 2 - 1;

const n = Math.random() * 400;

const classify = (x: number, y: number) => {
  return m * x + n > y ? [1, 0] : [0, 1];
};

// const classify = (x: number, y: number) => {
//   return x * x + y * y > 200 * 200 ? 1 : -1;
// };

const dataset: Point[] = [];

for (let d = 0; d < 500; d++) {
  const x = Math.random() * 400;
  const y = Math.random() * 400;
  const label = classify(x, y);

  dataset.push({ x, y, label });
}

const l = new Layer(2, 2);

new DrawTraining(400, 400, dataset, l);
