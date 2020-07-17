def keras_Bilstm(num_words=10000,max_len=500):
    embedding_dim = 16
    batch_size = 128
    model = tf.keras.models.Sequential([
        # input_dim为词汇表的大小  output_dim为输出embedding压缩后的维度   input_length为输入的长度
        keras.layers.Embedding(num_words, embedding_dim, input_length=max_len),
        # batch_size * max_length * embedding_dim
        #   -> batch_size * embedding_dim
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=64, return_sequences=True), ),
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=64, return_sequences=False), ),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, y_train,
                        epochs=30,
                        batch_size=batch_size,
                        validation_split=0.2)