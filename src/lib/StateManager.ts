class StateManager {
  static save(name: string, model: iModel) {
    const layers = model.layers.map((layer) => {
      const { weights, biases } = layer;

      return { weights, biases };
    });

    localStorage.setItem(name, JSON.stringify({ layers }));
  }

  static export(model: iModel) {
    const layers = model.layers.map((layer) => {
      const { weights, biases } = layer;

      return { weights, biases };
    });

    const blob = new Blob(JSON.stringify({ layers }).split(""));

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;
    a.download = "model.json";

    a.click();
  }

  static load(name: string, target: iModel) {
    const data = localStorage.getItem(name);

    if (data) {
      const model = JSON.parse(data) as iModel;

      model.layers.forEach(({ weights, biases }, id) => {
        target.layers[id].weights = weights;
        target.layers[id].biases = biases;
      });
    }
  }

  static import(file: File, target: iModel) {
    file.text().then((text) => {
      const model = JSON.parse(text) as iModel;

      model.layers.forEach(({ weights, biases }, id) => {
        target.layers[id].weights = weights;
        target.layers[id].biases = biases;
      });
    });
  }
}

export default StateManager;
