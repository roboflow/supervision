import tensorflow as tf
import numpy as np
import cv2


class OCR :
    def __init__(self, modelFile, labelFile):
        self.modelFile = modelFile
        self.labelFile = labelFile
        self.label = self.load_label(self.label_file) 
        self.graph = self.load_graph(self.model_file) 
        self.sess = tf.compat.v1.Session(graph=self.graph,  
                                         config=tf.compat.v1.ConfigProto())
        

    def load_graph(self, modelFile): 
           
        graph = tf.Graph() 
        graph_def = tf.compat.v1.GraphDef() 
           
        with open(modelFile, "rb") as f: 
            graph_def.ParseFromString(f.read()) 
           
        with graph.as_default(): 
            tf.import_graph_def(graph_def) 
           
        return graph 
    
    def load_label(self, labelFile): 
        label = [] 
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines() 
           
        for l in proto_as_ascii_lines: 
            label.append(l.rstrip()) 
           
        return label 
    
    def convert_tensor(self, image, imageSizeOuput): 
        """ 
        takes an image and transform it in tensor 
        """
        image = cv2.resize(image, 
                           dsize =(imageSizeOuput, 
                                  imageSizeOuput), 
                           interpolation = cv2.INTER_CUBIC) 
           
        np_image_data = np.asarray(image) 
        np_image_data = cv2.normalize(np_image_data.astype('float'), 
                                      None, -0.5, .5, 
                                      cv2.NORM_MINMAX) 
           
        np_final = np.expand_dims(np_image_data, axis = 0) 
           
        return np_final 
    
    def label_image(self, tensor): 
   
        input_name = "import/input"
        output_name = "import/final_result"
   
        input_operation = self.graph.get_operation_by_name(input_name) 
        output_operation = self.graph.get_operation_by_name(output_name) 
   
        results = self.sess.run(output_operation.outputs[0], 
                                {input_operation.outputs[0]: tensor}) 
        results = np.squeeze(results) 
        labels = self.label 
        top = results.argsort()[-1:][::-1] 
           
        return labels[top[0]] 
    

    def label_image_list(self, listImages, imageSizeOuput): 
        plate = "" 
           
        for img in listImages: 
               
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput)) 
           
        return plate, len(plate)