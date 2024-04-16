import typer
from pathlib import Path
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, track
import srsly
from escriptorium_connector import EscriptoriumConnector
from PIL import Image
import getpass
import requests 
from kraken.lib import models
from kraken.lib.default_specs import RECOGNITION_HYPER_PARAMS
from kraken.lib.train import KrakenTrainer, RecognitionModel
from kraken.ketos.util import to_ptl_device
import torch
from datetime import date
from zipfile import ZipFile
from io import BytesIO
import io

app = typer.Typer()


def make_training_data(E: EscriptoriumConnector, documents:list, transcription_pk:int, training_data_path: Path):
    # helper funtion to create training and evaluation data
    # training_data = [{'image': os.path.join(settings.MEDIA_ROOT, lt['image']),
    #                       'text': lt['content'],
    #                       'baseline': lt['baseline'],
    #                       'boundary': lt['mask']} for lt in ground_truth[partition:]]

    # evaluation_data = [{'image': os.path.join(settings.MEDIA_ROOT, lt['image']),
    if (training_data_path / "training_data.json").exists():
        training_data = srsly.read_json(training_data_path / "training_data.json")
        return training_data
    else:
        training_data = []

        for document in documents:
            parts = E.get_document_parts(document.pk)
            for part in track(parts.results, description=f"Downloading {document.name}..."):
                # Skip downloading images without text transcription 
                # TODO Evaluate this decision for cases where no text is the correct answer
                has_text = False
            
                image_path = str(training_data_path / f"{part.filename}")
                # get lines
                lines = E.get_document_part_lines(document.pk, part.pk) 
                #get baseline and mask
                for line in lines.results: 
                    baseline = line.baseline
                    boundary = line.mask
                    
                    # get text transcription
                    part_line_transcription = E.get_document_part_line_transcription_by_transcription(document.pk, part.pk,line.pk,transcription_pk) 
                    if part_line_transcription:
                        #ignore if content is None or empty
                        if part_line_transcription.content is None or part_line_transcription.content == "":
                            continue
                        else:
                            has_text = True
                            text = part_line_transcription.content
                            training_data.append({'image': image_path,
                                                'text': text,
                                                'baseline': baseline,
                                                'boundary': boundary})
                            
                if has_text and not (training_data_path / f"{part.filename}").exists():
                    img_binary = E.get_document_part_image(document.pk, part.pk)
                    img = Image.open(io.BytesIO(img_binary))
                    # save image
                    img.save(str(training_data_path / f"{part.filename}"))
                    
                    # save transcription
                    transcription = E.download_part_alto_transcription(
                        document.pk, part.pk, transcription_pk
                    )
                    with ZipFile(io.BytesIO(transcription)) as z:
                        with z.open(z.namelist()[0]) as f:
                            transcription = f.read()
                            Path(
                                str(training_data_path / f"{part.filename}.xml")
                            ).write_bytes(transcription)

        return training_data


def train(training_data, model_dir='escriptorium_trainer/models/HTR-Araucania_XIX.mlmodel', models_path:Path=Path('models'), model_name:str='new_model'):
    # helper function to train a new model
    # model_dir = path to existing model for fine-tuning
    # based on https://gitlab.com/scripta/escriptorium/-/blob/develop/app/apps/core/tasks.py?ref_type=heads#L418
    partition = int(len(training_data) / 10)

    training_data = [d for d in training_data[partition:]]
    evaluation_data = [d for d in training_data[:partition]]

    load = model_dir
    reorder = 'R' # for RTL languages else 'L'
    if torch.cuda.is_available():
        device = 'cuda:0'
    # If running on Mac with silicon chip, use Metal Performance Shaders
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = 'cpu'
    accelerator, device = to_ptl_device(device)

    kraken_model = RecognitionModel(hyper_params=RECOGNITION_HYPER_PARAMS,
                                    output=str(models_path) + "/" + model_name,
                                    model=load,
                                    reorder=reorder,
                                    format_type=None,
                                    training_data=training_data,
                                    evaluation_data=evaluation_data,
                                    partition=partition,
                                    num_workers=4,
                                    load_hyper_parameters=True,
                                    repolygonize=False,
                                    resize='add')

    trainer = KrakenTrainer(accelerator=accelerator,
                            devices=device,
                            # max_epochs=,
                            # min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=False,
                            val_check_interval=1.0,
                            # deterministic=ctx.meta['deterministic'],
                            #callbacks=[FrontendFeedback(model, model_dir, document.pk)]
                            )
    trainer.fit(kraken_model)

    if kraken_model.best_epoch == -1:
        print('Model did not converge, please try again with more data.')
    else:
        best_version = str(models_path) + "/" + kraken_model.best_model
        print(kraken_model.best_metric)
        return best_version

# add optional argument for --clear-secrets
@app.callback(invoke_without_command=True)
def main(clear_secrets: bool = typer.Option(False), fine_tune: bool = typer.Option(False)):
    """
    🏋️ escriptorium-trainer 🏋️
    A CLI for training kraken models from data on an eScriptorium server.
    """
    if clear_secrets:
        if Path("secrets.json").exists():
            Path("secrets.json").unlink()
            print("🏋️ secrets.json cleared 🏋️")

    if Path("secrets.json").exists():
        secrets = srsly.read_json("secrets.json")
    else:
        secrets = {}
        secrets["ESCRIPTORIUM_URL"] = (
            input("Please enter your Escriptorium Url: ")
            or "https://escriptorium.pennds.org/"
        )
        secrets["ESCRIPTORIUM_USERNAME"] = (
            input("Please enter your Escriptorium Username: ") or "invitado"
        )
        secrets["ESCRIPTORIUM_PASSWORD"] = getpass.getpass(
            "Please enter your Escriptorium Password:"
        )
        srsly.write_json("secrets.json", secrets)

    # connect to escriptorium
    E = EscriptoriumConnector(
        secrets["ESCRIPTORIUM_URL"],
        secrets["ESCRIPTORIUM_USERNAME"],
        secrets["ESCRIPTORIUM_PASSWORD"],
    )
    # get list of projects
    # not using E.get_projects() because it fails
    projects = requests.get(f"{secrets['ESCRIPTORIUM_URL']}/api/projects", headers=E.http.headers )
    if projects.status_code == 200:
        projects = projects.json()
        project_results = projects['results']
        project_names = [p['name'] for p in project_results]
        for i, name in enumerate(project_names):
            print(
                f"[bold green_yellow]{i}[/bold green_yellow] [bold white]{name}[/bold white]"
            )
        project_name = typer.prompt("Please select a project for training")
        # if the user enters a number, use that to select the document
        if project_name.isdigit():
            proj_name = project_results[int(project_name)]['name']
            project_slug = project_results[int(project_name)]['slug']
           
            print(
                f"[bold green_yellow] 🏋️ Training with {proj_name}...[/bold green_yellow]"
            )
        else:
            project_slug = None

    # create folder for training data
    training_data_path = Path("training_data")
    if not training_data_path.exists():
        training_data_path.mkdir(parents=True,exist_ok=True)
    # delete files in training_data folder
    #for file in training_data_path.iterdir():
    #    file.unlink()

    # create folder for models
    models_path = Path("models")
    if not models_path.exists():
        models_path.mkdir(parents=True,exist_ok=True)

    # fetch all project images
    all_documents = E.get_documents()
    documents = [doc for doc in all_documents.results if doc.project == project_slug]
    # TODO if there's a permissions problem, we can filter on project["owner"] and project["shared_with_users"]
    #select transcription text to train on IMPORTANT!
    if documents:
        transcriptions = E.get_document_transcriptions(documents[0].pk)
        transcription_names = [t.name for t in transcriptions]
        for i, name in enumerate(transcription_names):
            print(
                f"[bold green_yellow]{i}[/bold green_yellow] [bold white]{name}[/bold white]"
            )
        selection = typer.prompt("Please select a transcription text to train on")
        # if the user enters a number, use that to select the document
        if selection.isdigit():
            transcription_pk = transcriptions[int(selection)].pk
            transcription_name = transcriptions[int(selection)].name
            print(
                f"[bold green_yellow] 🏋️ Using text from {transcription_name}...[/bold green_yellow]"
            )
        else:
            print("Please enter a number to select the transcription text to train on")

        training_data = make_training_data(E, documents,transcription_pk, training_data_path)
        srsly.write_json(training_data_path / "training_data.json", training_data)


    else:
        print("No documents found in this project")

    # train a new model or fine-tune an existing one
    if fine_tune:
        e_url = secrets["ESCRIPTORIUM_URL"]
        # if e_url does not end with a slash, add it
        if e_url[-1] != "/":
            e_url = e_url + "/"
        models = requests.get(f'{e_url}api/models/', headers=E.http.headers) 
        if models.status_code == 200:
            models = models.json()
            model_names = [m["name"] for m in models["results"]]
            for i, name in enumerate(model_names):
                print(
                    f"[bold green_yellow]{i}[/bold green_yellow] [bold white]{name}[/bold white]"
                )
            model_name = typer.prompt("Please select a model for training")
            # if the user enters a number, use that to select the document
            if model_name.isdigit():
                models_path = Path("models")
                if not models_path.exists():
                    models_path.mkdir(parents=True,exist_ok=True)
                model = model_names[int(model_name)]
                model_uri = [m["file"] for m in models["results"] if m["name"] == model][0]
                model_filename = model_uri.split("/")[-1]
                if not (models_path / model_filename).exists():
                    model_request = requests.get(model_uri, headers=E.http.headers)
                    if model_request.status_code == 200:
                        (models_path / model_filename).write_bytes(model_request.content)
                        
                        print(f"[bold green_yellow] 🏋️ Fine-tune {model}...[/bold green_yellow]")
                    else:
                        print(f"[bold red]Error {model_request.raise_for_status()}[/bold red]")
                
                rec_model_path = str(models_path / model_filename)
                model = models.load_any(rec_model_path)
                print(
                    f"[bold green_yellow] 🏋️ Fine-tune {model}...[/bold green_yellow]"
                )
        else:
            print(f"[bold red]Error {models.raise_for_status()}[/bold red]")
    
    if fine_tune:
        model = train(training_data, model_dir=rec_model_path)
    else:
        model = train(training_data, model_name=f'new_model-{date.today()}')
        print(model)
    # choose projects to use for training 
    # fetch data
    # train model
    # push model to escriptorium



if __name__ == "__main__":
    app()