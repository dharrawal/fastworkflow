"""The local intent model at run time: ``CommandRouter`` answers a context through the trained artifacts, with the
model pipeline, the label encoders and the "is it trained" check it reads. Moved out of the trainer
(``fastworkflow.train.model_pipeline_training``, which imports them from here) so that the runtime does not import the
training module (bd fix-bl97 step 7)."""

import os
from typing import ClassVar
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import fastworkflow
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Union
import pickle
import threading
from fastworkflow.utils.logging import logger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Inference-side cache: path -> (artefact identity, encoder). See
# `get_label_encoder` for why the inference path must not read the trainer's global label encoder.
_label_encoder_cache: dict[str, tuple[tuple[int, int], LabelEncoder]] = {}


_label_encoder_cache_lock = threading.Lock()


def get_label_encoder(filepath) -> LabelEncoder:
    """Return the encoder pickled at *filepath*, unpickling it at most once per version.

    Callers must bind the result to a local. Assigning it to the module-level
    ``label_encoder`` would reintroduce the cross-context decode this exists to
    prevent: the module global is rebound by any concurrent prediction for a
    different context, and `predict_batch` releases the GIL for tens of ms, so
    the decode after it could run against another context's label space -
    either raising "y contains previously unseen labels" or silently returning
    another context's command name. Also spares every prediction an unpickle
    from disk, which the previous ``load_label_encoder`` call per prediction
    paid. The cache key is the artefact's (st_mtime_ns, st_size), so a retrain
    in the same process is not served a stale encoder.
    """
    stat = os.stat(filepath)
    identity = (stat.st_mtime_ns, stat.st_size)

    with _label_encoder_cache_lock:
        cached = _label_encoder_cache.get(filepath)
        if cached is not None and cached[0] == identity:
            return cached[1]

    # Unpickled outside the lock so a cold cache does not serialise every context's
    # first prediction behind one disk read. Two threads racing here both produce a
    # correct encoder for this artefact, so whichever result is stored is right.
    with open(filepath, 'rb') as f:
        encoder = pickle.load(f)

    with _label_encoder_cache_lock:
        _label_encoder_cache[filepath] = (identity, encoder)
    return encoder


def get_route_layer_filepath_model(workflow_folderpath,model_name) -> str:
    command_routing_definition = fastworkflow.RoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory
    return os.path.join(
        cmddir.get_commandinfo_folderpath(workflow_folderpath),
        model_name
    )


class CommandRouter:
    _instances_cache: ClassVar[dict[str, "CommandRouter"]] = {}

    def __new__(cls, model_artifacts_folderpath: str):
        """Return a cached instance if we've already created a CommandRouter for *model_artifacts_folderpath*.
        The path is normalised (the '*' replacement) so logically identical paths map to the same key.
        This avoids re-reading JSON threshold files **and**, more importantly, re-building the underlying
        ModelPipeline with expensive model loading.
        """
        # Normalise the path in the same way __init__ will do so that the cache key matches.
        normalised_path = model_artifacts_folderpath.replace('*', GLOBAL_CONTEXT_FOLDER)
        cached = cls._instances_cache.get(normalised_path)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        cls._instances_cache[normalised_path] = instance
        return instance

    #: Serialises construction: ``__new__`` caches the instance before ``__init__`` has loaded the
    #: models, so the server's background warm-up and a first turn could both load one head.
    _build_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, model_artifacts_folderpath: str):
        with CommandRouter._build_lock:
            self._init_once(model_artifacts_folderpath)

    def _init_once(self, model_artifacts_folderpath: str):
        # Avoid re-initialising if we are returning a cached instance.
        if getattr(self, "_initialised", False):
            return
        if '*' in model_artifacts_folderpath:
            model_artifacts_folderpath = model_artifacts_folderpath.replace('*', GLOBAL_CONTEXT_FOLDER)

        # A context of a version trained in a one-model layout (``embedding`` or ``workflow``)
        # holds no models of its own: a marker file points at the workflow's one model and says
        # which kind it is. The layout is read from the folder, never from a setting, so the
        # runtime serves whatever was published.
        from fastworkflow.train import intent_model_layout

        if intent_model_layout.is_workflow_scope_folder(model_artifacts_folderpath):
            if (intent_model_layout.context_layout(model_artifacts_folderpath)
                    == intent_model_layout.EMBEDDING):
                self._init_embedding_layout(model_artifacts_folderpath)
            else:
                self._init_workflow_layout(model_artifacts_folderpath)
            self._initialised = True
            return

        self.tiny_path = f"{model_artifacts_folderpath}/tinymodel.pth"
        self.large_path = f"{model_artifacts_folderpath}/largemodel.pth"
        threshold_path = f"{model_artifacts_folderpath}/threshold.json"
        tiny_ambiguous_threshold_path = f"{model_artifacts_folderpath}/tiny_ambiguous_threshold.json"
        large_ambiguous_threshold_path = f"{model_artifacts_folderpath}/large_ambiguous_threshold.json"
        self.label_encoder_path = f"{model_artifacts_folderpath}/label_encoder.pkl"
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            self.confidence_threshold = data['confidence_threshold']
            
        with open(tiny_ambiguous_threshold_path, 'r') as f:
            data = json.load(f)
            self.tiny_ambiguous_confidence_threshold = data['confidence_threshold']
        with open(large_ambiguous_threshold_path, 'r') as f:
            data = json.load(f)
            self.large_ambiguous_confidence_threshold = data['confidence_threshold']

        self.modelpipeline = ModelPipeline(
            tiny_model_path=self.tiny_path,
            distil_model_path=self.large_path,
            confidence_threshold=self.confidence_threshold
        )

        self._initialised = True

    def _init_workflow_layout(self, context_folderpath: str):
        """Read this context through the workflow's one large model (the ``workflow`` layout).

        The model is loaded once per artifact version and shared by every context of it
        (`WorkflowModelPipeline`, keyed by the real path of the version's ``workflow`` folder),
        so a context costs a mask and a threshold, not a model. The attributes a caller reads
        on a router keep their names: ``modelpipeline`` exposes the large model as
        ``distil_model`` / ``distil_tokenizer`` / ``device``, and
        ``label_encoder_path`` is the workflow's one encoder (every label, not this context's).
        """
        # Lazy: workflow_model imports this module.
        from fastworkflow.train import intent_model_layout, workflow_model

        marker = intent_model_layout.read_scope_marker(context_folderpath)
        model_dir = intent_model_layout.workflow_model_dir(context_folderpath)
        router = workflow_model.WorkflowModelRouter(model_dir, marker["context"], large_only=True)
        self._workflow_router = router
        self.tiny_path = None  # there is no tiny model in this layout
        self.large_path = os.path.join(model_dir, intent_model_layout.LARGE_MODEL_DIRNAME)
        self.label_encoder_path = router.label_encoder_path
        self.confidence_threshold = router.confidence_threshold
        self.tiny_ambiguous_confidence_threshold = router.tiny_ambiguous_confidence_threshold
        self.large_ambiguous_confidence_threshold = router.large_ambiguous_confidence_threshold
        self.modelpipeline = router.modelpipeline

    def _init_embedding_layout(self, context_folderpath: str):
        """Read this context through the workflow's embedding head (the ``embedding`` layout).

        The head is loaded once per artifact version and the encoder once per process
        (`embedding_head.EmbeddingHeadPipeline`, `SentenceEncoder`), so a context costs a mask.
        ``modelpipeline`` is the head's pipeline. There is no tiny or large model and no label
        encoder file: the label order is in the model folder's ``scopes.json``.
        """
        # Lazy: embedding_head imports this module.
        from fastworkflow.train import embedding_head, intent_model_layout

        marker = intent_model_layout.read_scope_marker(context_folderpath)
        model_dir = intent_model_layout.workflow_model_dir(context_folderpath)
        router = embedding_head.EmbeddingHeadRouter(model_dir, marker["context"])
        self._workflow_router = router
        self.tiny_path = None
        self.large_path = None
        self.label_encoder_path = None
        self.confidence_threshold = intent_model_layout.WORKFLOW_LAYOUT_TIER_THRESHOLD
        self.tiny_ambiguous_confidence_threshold = router.threshold
        self.large_ambiguous_confidence_threshold = router.threshold
        self.modelpipeline = router.modelpipeline

    def predict(self, command: str) -> list[str]:
        """
        if we are confident we will return a single label otherwise we will return a list
        """
        return self.predict_with_details(command)[0]

    def predict_with_details(self, command: str) -> tuple[list[str], dict]:
        """``predict`` plus the numbers behind the decision, for the
        ``fw.nlu.intent`` span attributes.

        The details dict is JSON-safe: {model_tier, confidence,
        ambiguous_threshold, confident, top_label, topk_labels, topk_scores}.
        The behavior of ``predict`` is unchanged — it delegates here.

        ``topk_scores`` is positionally aligned with ``topk_labels``. It is the
        probability behind each ranked label, which is what a top-k margin is
        the difference of; the winning label's own probability is
        ``confidence``, so the first score is that same number rather than a
        second measurement of it.

        A context of a one-model layout answers through the workflow's one model: the same
        keys (``model_tier`` is "embedding" for the embedding head and "large" for the
        fine-tuned model), and ``p_in``, ``p_out`` and ``escalation_margin``, the numbers
        behind "this request is not mine". Declining is
        the single label ``wildcard`` and a bare value the single label ``parameter_value``,
        as a per-context head reports them.
        """
        workflow_router = getattr(self, "_workflow_router", None)
        if workflow_router is not None:
            labels, details = workflow_router.predict_with_details(command)
            # `decision` restates what the labels already say; the details keep the documented keys.
            details.pop("decision", None)
            return labels, details

        results = predict_single_sentence(self.modelpipeline, command, self.label_encoder_path)
        used_distil = bool(results['used_distil'])
        ambiguous_threshold = (
            self.large_ambiguous_confidence_threshold
            if used_distil
            else self.tiny_ambiguous_confidence_threshold
        )
        confident = results['confidence'] > ambiguous_threshold
        labels = [results['label']] if confident else results['topk_labels']
        details = {
            "model_tier": "large" if used_distil else "tiny",
            "confidence": float(results['confidence']),
            "ambiguous_threshold": float(ambiguous_threshold),
            "confident": confident,
            "top_label": str(results['label']),
            "topk_labels": [str(label) for label in results['topk_labels']],
            "topk_scores": [float(score) for score in results['topk_scores']],
        }
        return list(labels), details


def warm_intent_heads(workflow_folderpath: str) -> list[str]:
    """Load every trained intent head of the workflow, and run one prediction on each, once at server start,
    so the first turn does not pay for it (``CommandRouter`` builds a ``ModelPipeline``, which loads two
    checkpoints and their tokenizers, ~0.5-1 s per context; the first prediction also reads the label encoder).

    Loads what the first turn would: the process-wide ``CommandRouter`` cache keyed by
    ``<folderpath>/___command_info/<context folder>`` (``*`` is the ``global`` folder), the keys the intent
    detection and gathering paths use, so *workflow_folderpath* must be the one the turn path passes, the
    ``Workflow``'s resolved ``folderpath``. A folder without a trained head (no tinymodel, label encoder and
    threshold, and not a context of the ``workflow`` layout, whose one model is loaded by the first context
    warmed and shared by the rest) is skipped. Returns the context folder names warmed. Never raises: a failure is logged and the
    turn path meets it again.
    """
    warmed: list[str] = []
    try:
        root = os.path.join(workflow_folderpath, "___command_info")
        if not os.path.isdir(root):
            return warmed
        from fastworkflow.train.intent_model_layout import workflow_head_present

        markers = ("tinymodel.pth", "label_encoder.pkl", "threshold.json")
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if name in ("versions", "current"):
                continue
            if not (all(os.path.exists(os.path.join(folder, m)) for m in markers)
                    or workflow_head_present(folder)):
                continue
            try:
                CommandRouter(folder).predict_with_details("warm up")
                warmed.append(name)
            except Exception as error:  # noqa: BLE001 - warm-up must not stop startup
                logger.warning(f"intent head warm-up failed for {folder}: {type(error).__name__}: {error}")
    except Exception as error:  # noqa: BLE001 - warm-up must not stop startup
        logger.warning(f"intent head warm-up failed for {workflow_folderpath}: {type(error).__name__}: {error}")
    return warmed


class ModelPipeline:
    # ------------------------------------------------------------------
    # Singleton-like caching ------------------------------------------------
    # ------------------------------------------------------------------
    _instances_cache: ClassVar[dict[tuple[str, str, float, str], "ModelPipeline"]] = {}

    def __new__(cls,
                tiny_model_path: str,
                distil_model_path: str,
                confidence_threshold: float = 0.65,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        key = (tiny_model_path, distil_model_path, confidence_threshold, device)
        existing = cls._instances_cache.get(key)
        if existing is not None:
            return existing
        instance = super().__new__(cls)
        cls._instances_cache[key] = instance
        return instance

    def __init__(
        self,
        tiny_model_path: str,
        distil_model_path: str,
        confidence_threshold: float = 0.65,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # __init__ will be called every time __new__ returns an instance – including
        # when we served a cached instance.  Guard against double-initialisation.
        if getattr(self, "_initialised", False):
            return

        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load TinyBERT
        self.tiny_tokenizer = AutoTokenizer.from_pretrained(tiny_model_path)
        self.tiny_model = AutoModelForSequenceClassification.from_pretrained(
            tiny_model_path
        ).to(device)

        # Load DistilBERT
        self.distil_tokenizer = AutoTokenizer.from_pretrained(distil_model_path)
        self.distil_model = AutoModelForSequenceClassification.from_pretrained(
            distil_model_path
        ).to(device)

        # Set models to evaluation mode
        self.tiny_model.eval()
        self.distil_model.eval()

        # Determine top-k value once for this pipeline (≤3, but never > num_labels)
        num_labels = self.tiny_model.config.num_labels
        self.k_val = min(3, num_labels)

        self._initialised = True

    def calculate_ndcg_at_k(self, batch_top_k_preds: List[List[int]], batch_top_k_scores: List[List[float]], true_labels: List[int], k: int = 3) -> float:
        batch_ndcg = 0.0
        
        for pred_top_k, conf_top_k, true_label in zip(batch_top_k_preds, batch_top_k_scores, true_labels):
            # Calculate relevance for top k predictions (1 if correct, 0 if incorrect)
            relevance = [1 if pred == true_label else 0 for pred in pred_top_k]
            
            # Calculate DCG
            dcg = 0.0
            for i in range(min(k, len(pred_top_k))):
                if relevance[i] == 1:
                    dcg += 1 / torch.log2(torch.tensor(i + 2, dtype=torch.float32))
            
            # Calculate IDCG (always 1/log2(2) since we only have one relevant document)
            idcg = 1 / torch.log2(torch.tensor(2, dtype=torch.float32))
            
            # Calculate NDCG for this sample
            sample_ndcg = dcg / idcg if idcg != 0 else 0
            batch_ndcg += sample_ndcg
            
        # Return average NDCG for the batch
        return batch_ndcg / len(true_labels)

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        k_val: int | None = None
    ) -> Dict:
        all_predictions = []
        all_confidences = []
        all_top_k_predictions = []  # Store top k predictions for each sample
        all_top_k_scores = []      # Store top k confidence scores for each sample
        all_logits = []
        all_used_distil = []
        k = k_val or self.k_val

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Predict with TinyBERT
            tiny_inputs = self.tiny_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            tiny_outputs = self.tiny_model(**tiny_inputs)
            tiny_logits = tiny_outputs.logits
            tiny_probs = torch.softmax(tiny_logits, dim=1)
            tiny_confidence, tiny_predictions = torch.max(tiny_probs, dim=1)

            # Get top k predictions and scores for TinyBERT
            tiny_top_k_scores, tiny_top_k_preds = torch.topk(tiny_probs, k=k, dim=1)

            # Identify low-confidence samples
            need_distil = tiny_confidence < self.confidence_threshold

            # Initialize with TinyBERT results
            batch_predictions = tiny_predictions.clone()
            batch_confidences = tiny_confidence.clone()
            batch_logits = tiny_logits.clone()
            batch_used_distil = need_distil.clone()
            batch_top_k_preds = tiny_top_k_preds.clone()
            batch_top_k_scores = tiny_top_k_scores.clone()

            # Predict with DistilBERT for low-confidence samples
            if need_distil.any():
                distil_texts = [text for text, flag in zip(batch_texts, need_distil) if flag]

                distil_inputs = self.distil_tokenizer(
                    distil_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                distil_outputs = self.distil_model(**distil_inputs)
                distil_logits = distil_outputs.logits
                distil_probs = torch.softmax(distil_logits, dim=1)
                distil_confidence, distil_predictions = torch.max(distil_probs, dim=1)

                # Get top k predictions and scores for DistilBERT
                distil_top_k_scores, distil_top_k_preds = torch.topk(distil_probs, k=k, dim=1)

                # Update results for low-confidence samples
                distil_idx = 0
                for j in range(len(batch_predictions)):
                    if need_distil[j]:
                        batch_predictions[j] = distil_predictions[distil_idx]
                        batch_confidences[j] = distil_confidence[distil_idx]
                        batch_logits[j] = distil_logits[distil_idx]
                        batch_top_k_preds[j] = distil_top_k_preds[distil_idx]
                        batch_top_k_scores[j] = distil_top_k_scores[distil_idx]
                        distil_idx += 1

            # Store results
            all_predictions.extend(batch_predictions.cpu().tolist())
            all_confidences.extend(batch_confidences.cpu().tolist())
            all_logits.append(batch_logits.cpu())
            all_used_distil.extend(batch_used_distil.cpu().tolist())
            all_top_k_predictions.extend(batch_top_k_preds.cpu().tolist())
            all_top_k_scores.extend(batch_top_k_scores.cpu().tolist())

        return {
            "predictions": all_predictions,
            "confidences": all_confidences,
            "logits": torch.cat(all_logits, dim=0).to(self.device),
            "used_distil": all_used_distil,
            "top_k_predictions": all_top_k_predictions,
            "top_k_scores": all_top_k_scores
        }

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, Dict]:
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_logits = []
        all_top_k_predictions = []
        all_top_k_scores = []
        total_used_distil = 0
        total_samples = 0
        total_ndcg = 0.0
        num_batches = 0

        for batch in tqdm(test_loader, desc="Evaluating"):
            # Batch now comes with raw *texts* so we can skip expensive decode→encode
            encodings, labels, texts = batch

            results = self.predict_batch(texts)

            all_predictions.extend(results['predictions'])
            all_labels.extend(labels.cpu().tolist())
            all_confidences.extend(results['confidences'])
            all_logits.append(results['logits'])
            all_top_k_predictions.extend(results['top_k_predictions'])
            all_top_k_scores.extend(results['top_k_scores'])
            total_used_distil += sum(results['used_distil'])
            total_samples += len(labels)

            # Calculate NDCG@3 for current batch
            batch_ndcg = self.calculate_ndcg_at_k(
                results['top_k_predictions'],
                results['top_k_scores'],
                labels.cpu().tolist()
            )
            total_ndcg += batch_ndcg
            num_batches += 1

        # Calculate F1 Score
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        # Calculate average NDCG@3 across all batches
        avg_ndcg = total_ndcg / num_batches

        # Model usage stats
        stats = {
            "total_samples": total_samples,
            "distil_usage": total_used_distil,
            "distil_percentage": (total_used_distil / total_samples) * 100,
            "tiny_percentage": ((total_samples - total_used_distil) / total_samples) * 100
        }

        return f1, avg_ndcg, stats


#for single utterance prediction
def predict_single_sentence(
    pipeline: ModelPipeline,
    text: str,
    path: str,
    #label_encoder: LabelEncoder
) -> Dict[str, Union[int, str, float, bool]]:

    # Input validation
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    if not text.strip():
        raise ValueError("Input text cannot be empty")


    # `path` is expected to be the absolute path to the label_encoder artefact
    encoder = get_label_encoder(path)
    k_val=len(encoder.classes_)
    k_val = 3 if k_val>2 else 2
    # Make prediction using the pipeline's batch prediction method
    results = pipeline.predict_batch([text],k_val=k_val)
    # Get the numeric prediction
    numeric_prediction = results["predictions"][0]

    label_names = encoder.inverse_transform(results['top_k_predictions'][0])

    # Convert numeric prediction back to original label name
    label_name = encoder.inverse_transform([numeric_prediction])[0]

    # `topk_scores` rides along with the labels it belongs to. `predict_batch`
    # already computed it in the same forward pass, and dropping it here was what
    # made `classifier-topk-margin` uncomputable downstream (FW-REQ-021 clause 1):
    # with only the labels, there is no second-best probability to subtract, and a
    # second forward pass to recover one would be a real performance change.
    # Ordering matches `topk_labels` — both come from the same `torch.topk`.
    return {
        "prediction": numeric_prediction,
        "label": label_name,
        "confidence": results["confidences"][0],
        "used_distil": results["used_distil"][0],
        "topk_labels":label_names,
        "topk_scores": results["top_k_scores"][0],
    }


# ---------------------------------------------------------------------
# Helper utilities for artefact locations (per-context)
# ---------------------------------------------------------------------
GLOBAL_CONTEXT_FOLDER = "global"


def is_workflow_trained(workflow_folderpath: str) -> Tuple[bool, List[str]]:
    """Return ``(is_trained, missing_context_folders)`` for *workflow_folderpath*.

    A workflow is considered trained when every routing context declared in its
    ``___command_info/routing_definition.json`` has a ``threshold.json`` model
    artifact under ``___command_info/<context>/`` (the special context ``"*"``
    maps to ``GLOBAL_CONTEXT_FOLDER``).

    This is a lightweight, filesystem-only check used to fail fast *before*
    starting a chat session on an untrained workflow. Without it, intent
    detection crashes on the first command when ``CommandRouter`` cannot open
    the missing ``threshold.json`` (e.g. ``___command_info/global/threshold.json``).
    """
    command_info_root = os.path.join(workflow_folderpath, "___command_info")
    routing_def_path = os.path.join(command_info_root, "routing_definition.json")
    if not os.path.isfile(routing_def_path):
        return False, ["<workflow not built>"]

    try:
        with open(routing_def_path, "r") as f:
            routing_definition = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, ["<unreadable routing_definition.json>"]

    contexts = routing_definition.get("contexts") or {}
    if not contexts:
        return False, ["<no contexts in routing_definition.json>"]

    # Only check the contexts that training actually produces artifacts for. This
    # mirrors the context selection in train(): the internal
    # command_metadata_extraction contexts (e.g. IntentDetection, ErrorCorrection)
    # are NOT trained per app workflow - their models live in the internal CME
    # workflow - and the global wildcard context "*" is always included.
    try:
        internal_wf_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
        internal_contexts = set(
            fastworkflow.CommandContextModel.load(internal_wf_path)._command_contexts.keys()
        )
    except Exception:
        internal_contexts = set()

    contexts_to_check = (set(contexts.keys()) - internal_contexts) | {"*"}

    missing: List[str] = []
    for context_name in contexts_to_check:
        context_folder = GLOBAL_CONTEXT_FOLDER if context_name == "*" else context_name
        threshold_path = os.path.join(command_info_root, context_folder, "threshold.json")
        if not os.path.isfile(threshold_path):
            missing.append(context_folder)

    return (not missing), sorted(missing)


def calculate_ndcg_at_k(logits, true_labels, k=3):
    probs = torch.softmax(logits, dim=1)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)
    batch_size = logits.shape[0]
    relevance = torch.zeros_like(top_k_probs)
    for i in range(batch_size):
        relevance[i] = (top_k_indices[i] == true_labels[i]).float()
    discounts = 1 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32)).to(device)
    dcg = torch.sum(relevance * discounts, dim=1)
    idcg = discounts[0]
    ndcg = dcg / idcg
    return ndcg.mean().item()
