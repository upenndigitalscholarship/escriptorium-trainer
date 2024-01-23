# WOrk from blog
https://digitalorientalist.com/2023/09/26/train-your-own-ocr-htr-models-with-kraken-part-1/
find training_data/*.xml > output.txt

ketos train -f xml /home/apjanco/Downloads/hi/*.xml

Errors
train.py:438
WARNING  No boundary given for line 
 or 

 ValueError: No valid training data was provided to the train command. Please add valid XML, line, or binary data.
---

training in eScriptorium
https://gitlab.com/scripta/escriptorium/-/blob/develop/app/apps/core/tasks.py?ref_type=heads#L418

qs = (LineTranscription.objects
              .filter(transcription=transcription,
                      line__document_part__pk__in=part_pks)
              .exclude(Q(content='') | Q(content=None)))

lines = E.get_document_part_lines(25,968) get baseline and mask
image = E.get_document_part_image(25,968) get image
# Need pathlike string, so save binary to disk and return path

text = E.get_document_part_line_transcription_by_transcription(25,968 ,10637,51) #doc, part, line, transcription 
text = text.content

RecognitionModel
https://github.com/mittagessen/kraken/blob/95981e0bcd354f37e2df7d3d07d40ebefc426400/kraken/lib/train.py#L199

1.
hyper_params: Dict[str, Any] = None,
from kraken.lib.default_specs import RECOGNITION_HYPER_PARAMS
hyper_params=RECOGNITION_HYPER_PARAMS,

output: str = 'model',
spec: str = default_specs.RECOGNITION_SPEC,
append: Optional[int] = None,
model: Optional[Union['PathLike', str]] = None,
reorder: Union[bool, str] = True,
2.
training_data: Union[Sequence[Union['PathLike', str]], Sequence[Dict[str, Any]]] = None,
training_data = [{'image': os.path.join(settings.MEDIA_ROOT, lt['image']),
                      'text': lt['content'],
                      'baseline': lt['baseline'],
                      'boundary': lt['mask']} for lt in ground_truth[partition:]]

3.
evaluation_data: Optional[Union[Sequence[Union['PathLike', str]], Sequence[Dict[str, Any]]]] = None,
evaluation_data = [{'image': os.path.join(settings.MEDIA_ROOT, lt['image']),
                        'text': lt['content'],
                        'baseline': lt['baseline'],
                        'boundary': lt['mask']} for lt in ground_truth[:partition]]

partition: Optional[float] = 0.9,
binary_dataset_split: bool = False,
num_workers: int = 1,
load_hyper_parameters: bool = False,
repolygonize: bool = False,
force_binarization: bool = False,
4.
format_type: Literal['path', 'alto', 'page', 'xml', 'binary'] = 'path',
codec: Optional[Dict] = None,
resize: Literal['fail', 'both', 'new', 'add', 'union'] = 'fail')

kraken_model = RecognitionModel(hyper_params=RECOGNITION_HYPER_PARAMS,
                                    output=os.path.join(model_dir, 'version'),
                                    # spec=spec,
                                    # append=append,
                                    model=load,
                                    reorder=reorder,
                                    format_type=None,
                                    training_data=training_data,
                                    evaluation_data=evaluation_data,
                                    partition=partition,
                                    # binary_dataset_split=fixed_splits,
                                    num_workers=LOAD_THREADS,
                                    load_hyper_parameters=True,
                                    repolygonize=False,
                                    # force_binarization=force_binarization,
                                    # codec=codec,
                                    resize='add')

TODO experiment, run training with RecognitionModel in shell 

class KrakenTrainer(pl.Trainer):
    def __init__(self,
                 enable_progress_bar: bool = True,
                 enable_summary: bool = True,
                 min_epochs: int = 5,
                 max_epochs: int = 100,
                 freeze_backbone=-1,
                 pl_logger: Union[pl.loggers.logger.Logger, str, None] = None,
                 log_dir: Optional['PathLike'] = None,
                 *args,
                 **kwargs):