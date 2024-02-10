from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty

from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton

from kivymd.font_definitions import theme_font_styles  

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class PlagiarismApp(MDApp):
    output = ObjectProperty(None)

    def generate_data(self, instance):
        sample_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
        sample_contents = []
        for file in sample_files:
            with open(file, 'r', encoding='utf-8') as f:
                sample_contents.append(f.read())

        vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
        similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

        vectors = vectorize(sample_contents)
        s_vectors = list(zip(sample_files, vectors))

        results = "Results show:\n\n"
        for sample_a, text_vector_a in s_vectors:
            new_vectors = s_vectors.copy()
            current_index = new_vectors.index((sample_a, text_vector_a))
            del new_vectors[current_index]
            for sample_b, text_vector_b in new_vectors:
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                sim_score_percentage = sim_score * 100
                results += f"{sample_a} and {sample_b} is {sim_score_percentage:.2f}% similar\n"
        
        self.output.text = results

    def build(self):
        layout = BoxLayout(orientation='vertical', padding=40, spacing=20)
        
        self.output = TextInput(
            hint_text='[File 1] and [File 2] is [%] similar ', 
            readonly=True, 
            multiline=True, 
            halign='center',
            font_size=24  
        )  

        generate_button = MDRaisedButton(
            text='Generate', 
            on_press=self.generate_data, 
            size_hint=(None, None), 
            size=(150, 50),
            pos_hint={'center_x': 0.5},  
            font_size=24  
        )

        layout.add_widget(self.output)
        layout.add_widget(generate_button)
        
        theme_font_styles.append('Roboto=fonts/Roboto-Regular.ttf')  
        
        return layout
    
    def on_start(self):
        self.title = "Plagiarism Detector"  

if __name__ == '__main__':
    PlagiarismApp().run()
