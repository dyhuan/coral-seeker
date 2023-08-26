import keras
import keras.models as models
import keras.layers as layers
import harness

def generate_model(conv2d_layers, conv2d_filters, dropout, dense_layer_size, concatenate):
    input = keras.Input(shape=(harness.IMG_SIZE, harness.IMG_SIZE, harness.channels), name="img")

    x = input
    convs = []

    for i in range(conv2d_layers):
        x = layers.Conv2D(conv2d_filters, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        convs.append(x)

    if concatenate:
        x = layers.concatenate(convs)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_layer_size, activation='relu')(x)

    output = layers.Dense(len(harness.corals), activation='sigmoid')(x)

    model = keras.Model(input, output, name="coral_model")

    print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    harness.train_model(model, 100, 10)
    # harness.test_model_arbitrary(model)
    return harness.test_model(model)
    

# we will vary the following parameters:
# # of conv2d + pooling layers (1-3)
# # of conv2d filters (16-64)
# dropout (0.25-0.75)
# dense layer size (256-2048)
def generate_models():
    with open("results.csv", "w") as results_file:
        for conv2d_layers in range(1, 4):
            for conv2d_filters in range(16, 65, 16):
                for dropout in range(25, 76, 25):
                    for dense_layer_size in range(256, 1025, 256):
                        for concatenate in [True, False]:
                            results = generate_model(conv2d_layers, conv2d_filters, dropout / 100, dense_layer_size, concatenate)
                            results_file.write(f"{conv2d_layers},{conv2d_filters},{dropout / 100},{dense_layer_size},{concatenate},{results[0]},{results[1]}\n")

generate_models()
