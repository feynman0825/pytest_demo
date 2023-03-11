from pytest_demo.model import LinearModel
from pytest_demo.process import etl, get_data


def go():
    path = "pytest_demo/data/data.npy"
    data = get_data(path)
    X_train = data[:-1, :-1]
    y_train = data[:-1, -1]
    X_test = data[-1:, :-1]
    y_test = data[-1:, -1]

    processed_X_train = etl(X_train)
    processed_X_test = etl(X_test)

    model = LinearModel()
    model.train(processed_X_train, y_train)
    print(f"Model perametesr: {model.beta}")

    pred = model.predict(processed_X_test)
    print(f"Model predict: {pred}")

    print(f"Actual targets: {y_test}")

if __name__ == "__main__":
    go()
