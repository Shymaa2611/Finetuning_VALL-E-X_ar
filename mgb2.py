from bs4 import BeautifulSoup
from pydub import AudioSegment
import os

def parse_xml(xml_file):
    segments = []
    with open(xml_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
        for segment in soup.find_all('segment'):
            segment_data = {}
            segment_data['id'] = segment['id']
            segment_data['starttime'] = float(segment['starttime'])
            segment_data['endtime'] = float(segment['endtime'])
            segment_data['who'] = segment['who']
            words = [element.text for element in segment.find_all('element')]
            segment_data['words'] = ' '.join(words)
            segments.append(segment_data)
    return segments

def split_audio_segment(input_file, start, end, dataset_part, count):
    sound = AudioSegment.from_file(input_file)
    start_ms = start * 1000
    end_ms = end * 1000
    segment = sound[start_ms:end_ms]
    os.makedirs(os.path.join(f"mgb2_dataset/{dataset_part}/wav"), exist_ok=True)
    segment.export(os.path.join(f"mgb2_dataset/{dataset_part}/wav", f"wav_{count+1}.wav"), format="wav")

def create_mgb2_dataset(dataset_part, xml_utf8, wav_dir):
    try:
        os.makedirs(os.path.join("mgb2_dataset", dataset_part, "txt"), exist_ok=True)
        print("Text directory created successfully.")
        os.makedirs(os.path.join("mgb2_dataset", dataset_part, "wav"), exist_ok=True)
        print("WAV directory created successfully.")
        count = 0
        xml_files = sorted(os.listdir(xml_utf8))
        for xml_file in xml_files:
            segments = parse_xml(os.path.join(xml_utf8, xml_file))
            
            for segment in segments:
                words = segment['words']
                file_path = os.path.join("mgb2_dataset", dataset_part, "txt", f"text_{count+1}.txt")
                with open(file_path, 'w', encoding='utf-8') as txtfile:
                    txtfile.write(words)
                print(f"File '{file_path}' created successfully.")

                wav_file = os.path.splitext(xml_file)[0] + ".wav"
                start = int(segment['starttime']) 
                end = int(segment['endtime']) + 1
                split_audio_segment(os.path.join(wav_dir, wav_file), start, end, dataset_part, count)
                count += 1
    except FileExistsError:
        print("Directories already exist.")

def create_dataset_parts(dataset_dir):
    dataset_parts=["train","test","dev"]
    for part in dataset_parts:
        if part=="train":
             xml_utf8 = f"{dataset_dir}\\train\\xml\\utf8"
             wav_dir = f"{dataset_dir}\\train\\wav"
             create_mgb2_dataset(part,xml_utf8,wav_dir)
        elif part=="test":
             xml_utf8 = f"{dataset_dir}\\test\\xml\\utf8"
             wav_dir = f"{dataset_dir}\\test\\wav"
             create_mgb2_dataset(part,xml_utf8,wav_dir)
        else:
            xml_utf8 = f"{dataset_dir}\\dev\\xml\\utf8"
            wav_dir = f"{dataset_dir}\\dev\\wav"
            create_mgb2_dataset(part, xml_utf8, wav_dir)



if __name__ == "__main__":
    print("done!")

