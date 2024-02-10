from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class PlagiarismApp(App):
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

        results = ""
        for sample_a, text_vector_a in s_vectors:
            new_vectors = s_vectors.copy()
            current_index = new_vectors.index((sample_a, text_vector_a))
            del new_vectors[current_index]
            for sample_b, text_vector_b in new_vectors:
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                sim_score_percentage = sim_score * 100
                results += f"{sample_a} vs {sample_b}: {sim_score_percentage:.2f}%\n"
        
        self.output.text = results

    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        self.output = TextInput(
            hint_text='Output', 
            readonly=True, 
            multiline=True
        )  

        generate_button = Button(
            text='Generate', 
            on_press=self.generate_data
        )

        layout.add_widget(self.output)
        layout.add_widget(generate_button)
        
        return layout
    
if __name__ == '__main__':
    PlagiarismApp().run()
