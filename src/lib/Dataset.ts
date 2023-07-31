class Dataset extends Array<Data.Data> {
  public batchSize: number;

  constructor(batchSize: number) {
    super();

    this.batchSize = batchSize;
  }

  generate(g: Data.Generator, c: Data.Classifier) {
    for (let i = 0; i < this.batchSize; i++) {
      const value = g();

      const label = c(value);

      this.push({ value, label });
    }
  }

  shuffle() {
    const shuffled = [];

    while (this.length) {
      const index = Math.random() * (this.length - 1);

      const [item] = this.splice(Math.round(index), 1);

      shuffled.push(item);
    }

    this.push(...shuffled);
  }
}

export default Dataset;
