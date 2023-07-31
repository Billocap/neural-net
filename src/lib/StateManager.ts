class StateManager {
  static save(name: string, model: Model.Model) {
    const layers = model.layers.map((layer) => {
      const { weights, biases } = layer;

      return { weights, biases };
    });

    localStorage.setItem(name, JSON.stringify({ layers }));
  }

  static export(model: Model.Model) {
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

  static load(name: string, target: Model.Model) {
    const data = localStorage.getItem(name);

    if (data) {
      const model = JSON.parse(data) as Model.Model;

      model.layers.forEach(({ weights, biases }, id) => {
        target.layers[id].weights = weights;
        target.layers[id].biases = biases;
      });
    }
  }

  static import(file: File, target: Model.Model) {
    file.text().then((text) => {
      const model = JSON.parse(text) as Model.Model;

      model.layers.forEach(({ weights, biases }, id) => {
        target.layers[id].weights = weights;
        target.layers[id].biases = biases;
      });
    });
  }
}

export default StateManager;
