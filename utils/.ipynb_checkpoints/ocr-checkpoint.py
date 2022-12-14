import tensorflow as tf

SORTED_CHARS = sorted([' ', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
IMG_HEIGHT = 64
MAX_WIDTH = 1024
MAX_LENGTH = 25
char_to_num = tf.keras.layers.StringLookup(vocabulary=SORTED_CHARS, mask_token=None)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

class CTCDecodeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CTCDecodeLayer, self).__init__(**kwargs)
        self.decoder = tf.keras.backend.ctc_decode
    
    def __call__(self, y_pred):
        shape = tf.shape(y_pred)
        input_len = tf.ones(shape[0]) * tf.cast(shape[1], dtype=tf.float32)
        results = self.decoder(y_pred, input_length=input_len, greedy=True)[0][0][:,:MAX_LENGTH]
        results = tf.strings.reduce_join(num_to_char(results), axis=1)
        return results

class OcrEngine():
    def __init__(self, model_path, input_shape=(MAX_WIDTH, IMG_HEIGHT, 3)):
        self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CH = input_shape
        model = tf.keras.models.load_model(model_path, compile=False)
        image = tf.keras.layers.Input(shape=input_shape, name='Image')
        y_pred = model(image, training=False)
        decoded = CTCDecodeLayer(name='CTC_Decode')(y_pred)
        inference_model = tf.keras.Model(inputs=image, outputs=decoded)
        # warm up
        temp = tf.zeros((1, self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=tf.float32)
        temp = tf.transpose(temp, perm=[0, 2, 1, 3])
        _ = inference_model.predict(temp)
        
        self.inference_model = inference_model
        
    def _encode_single_img(self, img):
        h, w = img.shape[:2]
        img = tf.image.convert_image_dtype(img, tf.float32) # 0~255 -> 0~1
        target_h, target_w = self.IMG_HEIGHT, int(w/h*self.IMG_HEIGHT)
        img = tf.image.resize(img, [target_h, target_w])
        img = tf.image.pad_to_bounding_box(img, 0, 0, self.IMG_HEIGHT, self.IMG_WIDTH)
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    
    def __call__(self, date_img):
        date_img = self._encode_single_img(date_img)
        pred_str = self.inference_model.predict(date_img[None], verbose=0)[0]
        pred_str = pred_str.decode('utf-8').strip('[UNK]')
        return pred_str