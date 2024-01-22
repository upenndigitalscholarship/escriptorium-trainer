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
from zipfile import ZipFile
from io import BytesIO
import io

app = typer.Typer()


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
    projects = E.get_projects()
    project_names = [p.name for p in projects.results]
    for i, name in enumerate(project_names):
        print(
            f"[bold green_yellow]{i}[/bold green_yellow] [bold white]{name}[/bold white]"
        )
    project_name = typer.prompt("Please select a project for training")
    # if the user enters a number, use that to select the document
    if project_name.isdigit():
        project_pk = projects.results[int(project_name)].id
        project_slug = projects.results[int(project_name)].slug
        E.set_connector_project_by_pk(project_pk)
        print(
            f"[bold green_yellow] 🏋️ Training with {E.project_name}...[/bold green_yellow]"
        )
    else:
        project_slug = None

    # fetch all project images
    all_documents = E.get_documents()
    documents = [doc for doc in all_documents.results if doc.project == project_slug]
    for document in documents:
        transcriptions = E.get_document_transcriptions(document.pk)
        manual = [t for t in transcriptions if t.name == "manual"]
        if manual:
            transcription_pk = manual[0].pk

        parts = E.get_document_parts(document.pk)
        for part in track(parts.results, description=f"Downloading {document.name}..."):
            img_binary = E.get_document_part_image(document.pk, part.pk)
            img = Image.open(io.BytesIO(img_binary))
            alto_xml = E.download_part_alto_transcription(
                document.pk, part.pk, transcription_pk
            )
            # You will need to unzip these bytes in order to access the XML data (zipfile can do this).
            with ZipFile(BytesIO(alto_xml)) as z:
                with z.open(z.namelist()[0]) as f:
                    alto_xml = f.read()
            # create folder for training data
            training_data_path = Path("training_data")
            if not training_data_path.exists():
                training_data_path.mkdir(parents=True,exist_ok=True)
            # save image
            img.save(training_data_path / f"{document.name}_{part.name}.png")
            # save alto xml
            with open(training_data_path / f"{document.name}_{part.name}.xml", "wb") as f:
                f.write(alto_xml)
        
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
    # choose projects to use for training 
    # fetch data
    # train model
    # push model to escriptorium

if __name__ == "__main__":
    app()