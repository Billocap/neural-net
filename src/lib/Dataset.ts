class Dataset extends Array<iData> {
  public batchSize: number;

  constructor(batchSize: number) {
    super();

    this.batchSize = batchSize;
  }

  generate(g: iGenerator, c: iClassifier) {
    for (let i = 0; i < this.batchSize; i++) {
      const value = g();

      const label = c(value);

      this.push({ value, label });
    }
  }
}

export default Dataset;
