// Variáveis globais
let agRunning = false;
let fitnessChart;
let generationData = [];
let bestIndividuals = [];
let currentTaskId = null;
let intervalId = null;
let startTime = null;
let logIntervalId = null;
let elapsedInterval;
let overallBestAccuracy = 0;
let task_logs = {}; // Adicione esta linha

// Inicialização
document.addEventListener("DOMContentLoaded", function () {
  // Inicializar gráfico
  initFitnessChart();

  updateResultsTable();

  // Event listeners
  document
    .getElementById("mutation-rate")
    .addEventListener("input", function () {
      document.getElementById("mutation-rate-value").textContent = this.value;
    });

  document
    .getElementById("selection-method")
    .addEventListener("change", function () {
      const method = this.value;
      document.getElementById("elitism-size-container").style.display =
        method === "elitismo" ? "block" : "none";
      document.getElementById("tournament-size-container").style.display =
        method === "torneio" ? "block" : "none";
    });

  document.getElementById("start-ag").addEventListener("click", startAlgorithm);
});

function showToast(message, type = "info", duration = 3000) {
  Toastify({
    text: message,
    duration: duration,
    close: true,
    gravity: "top",
    position: "center",
    className: `toastify-${type}`,
    stopOnFocus: true,
  }).showToast();
}

async function updateClassificationExamples(taskId, startTime = Date.now()) {
  const MAX_DURATION = 5 * 60 * 1000; // 5 minutos em milissegundos
  const RETRY_DELAY = 5000;

  const elapsed = Date.now() - startTime;
  const progress = Math.min(100, (elapsed / MAX_DURATION) * 100);

  const progressBar = document.querySelector(".loading-progress-bar");
  if (progressBar) {
    progressBar.style.width = `${progress}%`;
  }

  try {
    const correctContainer = document.getElementById("correct-classifications");
    const wrongContainer = document.getElementById("wrong-classifications");

    const tentativas = Math.floor(elapsed / RETRY_DELAY) + 1;

    const loadingHTML = `
  <div class="col-12 loading-container">
    <div class="loading-spinner"></div>
    <div class="loading-text">
      Buscando exemplos (Tempo estimado: 5min)...<br>
      <small>Tentativa ${tentativas}</small>
    </div>
    <div class="loading-progress mt-2">
      <div class="loading-progress-bar" style="width: ${progress}%"></div>
    </div>
  </div>
`;

    correctContainer.innerHTML = loadingHTML;
    wrongContainer.innerHTML = loadingHTML;

    const response = await fetch(
      `http://localhost:8000/sample_images/${taskId}`
    );

    if (!response.ok) {
      if (response.status === 404 && elapsed < MAX_DURATION) {
        const loadingText = document.querySelectorAll(".loading-text");
        if (loadingText) {
          loadingText.forEach((text) => {
            const tentativas = Math.floor(elapsed / RETRY_DELAY) + 1;
            text.innerHTML = `Buscando exemplos (Tempo estimado: 5min)...<br><small>Tentativa ${tentativas}</small>`;
          });
        }

        setTimeout(
          () => updateClassificationExamples(taskId, startTime),
          RETRY_DELAY
        );
        return;
      }

      const error = await response.json();
      throw new Error(error.detail || "Erro ao carregar exemplos");
    }

    const samples = await response.json();

    const classNames = [
      "avião",
      "carro",
      "pássaro",
      "gato",
      "cervo",
      "cachorro",
      "sapo",
      "cavalo",
      "navio",
      "caminhão",
    ];

    let correctHTML = "";
    let wrongHTML = "";

    samples.forEach((sample, idx) => {
      const containerHTML = `
  <div class="col-4 mb-4 classification-examples">
    <div class="card h-100 shadow">
      <div class="card-body text-center d-flex flex-column">
        <div class="mb-2">
          <button class="btn btn-sm btn-outline-light btn-dark view-btn mb-2" 
                  data-index="${idx}" data-bs-toggle="modal" data-bs-target="#imageModal">
            <i class="bi bi-eye-fill"></i> Visualizar Imagem
          </button>
        </div>
        <div class="mb-2 flex-grow-1 d-flex align-items-center justify-content-center">
          <canvas class="sample-image" width="32" height="32" data-index="${idx}"
                  style="width: 100%; max-width: 32px; image-rendering: pixelated;"></canvas>
        </div>
        <div>
          <p class="mb-1"><strong>Predição:</strong> ${
            classNames[sample.pred]
          }</p>
          <p class="mb-0"><strong>Verdadeiro:</strong> ${
            classNames[sample.true]
          }</p>
        </div>
      </div>
    </div>
  </div>
`;
      if (sample.correct) {
        correctHTML += containerHTML;
      } else {
        wrongHTML += containerHTML;
      }
    });

    correctContainer.innerHTML =
      correctHTML ||
      `<div class="col-12 text-center text-muted py-4">
        <i class="bi bi-info-circle"></i> Nenhum acerto disponível para exibição
      </div>`;

    wrongContainer.innerHTML =
      wrongHTML ||
      `<div class="col-12 text-center text-muted py-4">
        <i class="bi bi-info-circle"></i> Nenhum erro disponível para exibição
      </div>`;

    document.querySelectorAll(".sample-image").forEach((canvas) => {
      const ctx = canvas.getContext("2d");
      const sample = samples[canvas.dataset.index];
      const imgData = sample.image;

      const imageData = ctx.createImageData(32, 32);
      const data = imageData.data;

      for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 32; x++) {
          const idx = (y * 32 + x) * 4;
          const pixel = imgData[y][x];
          data[idx] = Math.floor(pixel[0] * 255);
          data[idx + 1] = Math.floor(pixel[1] * 255);
          data[idx + 2] = Math.floor(pixel[2] * 255);
          data[idx + 3] = 255;
        }
      }

      ctx.putImageData(imageData, 0, 0);

      // Botão "Ver maior"
      const btn = canvas.closest(".card-body").querySelector(".view-btn");
      btn.addEventListener("click", () => {
        const dataURL = canvas.toDataURL("image/png");
        const modalImage = document.getElementById("modalImage");
        modalImage.src = dataURL;
      });
    });
  } catch (error) {
    console.error("Erro ao carregar exemplos:", error);

    const errorHTML = `
      <div class="col-12 text-center py-4">
        <div class="text-danger">
          <i class="bi bi-exclamation-triangle-fill"></i>
          <h5>Erro ao carregar exemplos</h5>
          <p class="text-muted">${error.message}</p>
          <button class="btn btn-sm btn-outline-primary" onclick="updateClassificationExamples('${taskId}')">
            <i class="bi bi-arrow-clockwise"></i> Tentar novamente
          </button>
        </div>
      </div>
    `;

    document.getElementById("correct-classifications").innerHTML = errorHTML;
    document.getElementById("wrong-classifications").innerHTML = errorHTML;
  }
}

function initFitnessChart() {
  const ctx = document.getElementById("fitness-chart").getContext("2d");
  fitnessChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Melhor Acurácia por Geração",
          data: [],
          borderColor: "#6f42c1",
          backgroundColor: "rgba(111, 66, 193, 0.1)",
          borderWidth: 2,
          tension: 0.1,
          fill: true,
        },
        {
          label: "Média da População",
          data: [],
          borderColor: "#20c997",
          backgroundColor: "rgba(32, 201, 151, 0.1)",
          borderWidth: 2,
          tension: 0.1,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          min: 0,
          max: 1,
          ticks: {
            callback: function (value) {
              return (value * 100).toFixed(0) + "%";
            },
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function (context) {
              return (
                context.dataset.label +
                ": " +
                (context.raw * 100).toFixed(2) +
                "%"
              );
            },
          },
        },
      },
    },
  });
}

function startAlgorithm() {
  if (agRunning) {
    showToast("O algoritmo já está em execução!", "warning", 3000);
    return;
  }

  agRunning = true;
  document.getElementById("start-ag").disabled = true;
  document.getElementById("start-ag").innerHTML =
    '<i class="bi bi-hourglass"></i> Executando...';

  document.getElementById("stop-ag").disabled = false;

  // Resetar dados
  generationData = [];
  bestIndividuals = [];
  fitnessChart.data.labels = [];
  fitnessChart.data.datasets[0].data = [];
  fitnessChart.data.datasets[1].data = [];
  fitnessChart.update();
  document.getElementById("log-console").textContent = "";
  overallBestAccuracy = 0;
  document.getElementById("overall-best-accuracy").textContent = "0.00%";

  // Obter parâmetros
  const popSize = parseInt(document.getElementById("pop-size").value);
  const numGenerations = parseInt(
    document.getElementById("num-generations").value
  );
  const numEpochs = parseInt(document.getElementById("num-epochs").value);
  const mutationRate = parseFloat(
    document.getElementById("mutation-rate").value
  );
  const selectionMethod = document.getElementById("selection-method").value;
  const elitismSize = parseInt(document.getElementById("elitism-size").value);
  const tournamentSize = parseInt(
    document.getElementById("tournament-size").value
  );

  // Obter espaço de busca
  const space = {
    learning_rate: Array.from(
      document.querySelectorAll("#learning-rate option:checked")
    ).map((opt) => parseFloat(opt.value)),
    batch_size: Array.from(
      document.querySelectorAll("#batch-size option:checked")
    ).map((opt) => parseInt(opt.value)),
    n_filters: Array.from(
      document.querySelectorAll("#n-filters option:checked")
    ).map((opt) => parseInt(opt.value)),
    n_fc: Array.from(document.querySelectorAll("#n-fc option:checked")).map(
      (opt) => parseInt(opt.value)
    ),
    dropout: Array.from(
      document.querySelectorAll("#dropout option:checked")
    ).map((opt) => parseFloat(opt.value)),
    weight_decay: Array.from(
      document.querySelectorAll("#weight-decay option:checked")
    ).map((opt) => parseFloat(opt.value)),
    activation: Array.from(
      document.querySelectorAll("#activation option:checked")
    ).map((opt) => opt.value),
    optimizer: Array.from(
      document.querySelectorAll("#optimizer option:checked")
    ).map((opt) => opt.value),
    stride: [1], // Valor fixo
    aggregation: Array.from(
      document.querySelectorAll("#aggregation option:checked")
    ).map((opt) => opt.value),
  };

  // Atualizar UI
  document.getElementById("progress-text").textContent = `0/${numGenerations}`;
  document.getElementById("progress-bar").style.width = "0%";
  document.getElementById("overall-best-accuracy").textContent = "0.00%";
  document.getElementById("elapsed-time").textContent = "00:00";
  document.getElementById("best-accuracy").textContent = "Acurácia: -";

  // Configuração para enviar
  const config = {
    pop_size: popSize,
    geracoes: numGenerations,
    taxa_mutacao: mutationRate,
    metodo_selecao: selectionMethod,
    elite_size: elitismSize,
    tamanho_torneio: tournamentSize,
    n_epochs: numEpochs,
    space: space,
  };

  startTime = new Date();

  // Variável para armazenar o intervalo do cronômetro
  let elapsedInterval;

  // Função para atualizar o tempo decorrido
  function updateElapsedTime() {
    const elapsedMs = new Date() - startTime;
    const elapsedMinutes = Math.floor(elapsedMs / 60000);
    const elapsedSeconds = Math.floor((elapsedMs % 60000) / 1000);
    document.getElementById("elapsed-time").textContent = `${elapsedMinutes
      .toString()
      .padStart(2, "0")}:${elapsedSeconds.toString().padStart(2, "0")}`;
  }

  // Iniciar o cronômetro
  elapsedInterval = setInterval(updateElapsedTime, 1000);
  updateElapsedTime(); // Chamada inicial para evitar atraso de 1s

  // Enviar requisição para iniciar o AG
  fetch("http://localhost:8000/start_ag", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
      "Access-Control-Request-Headers": "content-type",
    },
    body: JSON.stringify(config),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.task_id) {
        currentTaskId = data.task_id;
        startLogPolling(currentTaskId);
        showToast(
          "O algoritmo iniciou! Você pode acompanhar o progresso e os logs no dashboard ao lado.",
          "success",
          3000
        );
        intervalId = setInterval(
          () => checkTaskStatus(currentTaskId, numGenerations),
          2000
        );
      } else {
        throw new Error("Falha ao iniciar o AG: " + JSON.stringify(data));
      }
    })
    .catch((error) => {
      console.error("Erro:", error);
      showToast(
        "Erro ao iniciar o algoritmo genético: " + error.message,
        "error",
        5000
      );

      // Limpar intervalos em caso de erro
      clearInterval(elapsedInterval);
      if (intervalId) clearInterval(intervalId);

      agRunning = false;
      document.getElementById("start-ag").disabled = false;
      document.getElementById("start-ag").innerHTML =
        '<i class="bi bi-play-fill"></i> Iniciar Algoritmo';
      document.getElementById("stop-ag").disabled = true;
    });

  // Adicionar listener para parar o cronômetro quando o botão de parar for clicado
  document.getElementById("stop-ag").addEventListener("click", function () {
    clearInterval(elapsedInterval);
  });
}

function stopAlgorithm(showToastMessage = true) {
  // Limpar todos os intervalos
  clearInterval(elapsedInterval);
  clearInterval(intervalId);
  clearInterval(logIntervalId);

  // Resetar estado
  agRunning = false;
  currentTaskId = null;

  // Resetar UI
  document.getElementById("start-ag").disabled = false;
  document.getElementById("start-ag").innerHTML =
    '<i class="bi bi-play-fill"></i> Iniciar Algoritmo';
  document.getElementById("stop-ag").disabled = true;
  document.getElementById("stop-ag").style.display = "none";
}

function startAlgorithm() {
  if (agRunning) {
    showToast("O algoritmo já está em execução!", "warning", 3000);
    return;
  }

  agRunning = true;
  document.getElementById("start-ag").disabled = true;
  document.getElementById("start-ag").innerHTML =
    '<i class="bi bi-hourglass"></i> Executando...';
  document.getElementById("stop-ag").style.display = "block";
  document.getElementById("stop-ag").disabled = false;

  // Resetar dados
  generationData = [];
  bestIndividuals = [];
  fitnessChart.data.labels = [];
  fitnessChart.data.datasets[0].data = [];
  fitnessChart.data.datasets[1].data = [];
  fitnessChart.update();
  document.getElementById("log-console").textContent = "";

  // Obter parâmetros
  const popSize = parseInt(document.getElementById("pop-size").value);
  const numGenerations = parseInt(
    document.getElementById("num-generations").value
  );
  const numEpochs = parseInt(document.getElementById("num-epochs").value);
  const mutationRate = parseFloat(
    document.getElementById("mutation-rate").value
  );
  const selectionMethod = document.getElementById("selection-method").value;
  const elitismSize = parseInt(document.getElementById("elitism-size").value);
  const tournamentSize = parseInt(
    document.getElementById("tournament-size").value
  );

  // Obter espaço de busca (código existente)
  const space = {
    learning_rate: Array.from(
      document.querySelectorAll("#learning-rate option:checked")
    ).map((opt) => parseFloat(opt.value)),
    batch_size: Array.from(
      document.querySelectorAll("#batch-size option:checked")
    ).map((opt) => parseInt(opt.value)),
    n_filters: Array.from(
      document.querySelectorAll("#n-filters option:checked")
    ).map((opt) => parseInt(opt.value)),
    n_fc: Array.from(document.querySelectorAll("#n-fc option:checked")).map(
      (opt) => parseInt(opt.value)
    ),
    dropout: Array.from(
      document.querySelectorAll("#dropout option:checked")
    ).map((opt) => parseFloat(opt.value)),
    weight_decay: Array.from(
      document.querySelectorAll("#weight-decay option:checked")
    ).map((opt) => parseFloat(opt.value)),
    activation: Array.from(
      document.querySelectorAll("#activation option:checked")
    ).map((opt) => opt.value),
    optimizer: Array.from(
      document.querySelectorAll("#optimizer option:checked")
    ).map((opt) => opt.value),
    stride: [1],
    aggregation: Array.from(
      document.querySelectorAll("#aggregation option:checked")
    ).map((opt) => opt.value),
  };

  // Atualizar UI
  document.getElementById("progress-text").textContent = `0/${numGenerations}`;
  document.getElementById("progress-bar").style.width = "0%";
  document.getElementById("overall-best-accuracy").textContent = "0.00%";
  document.getElementById("elapsed-time").textContent = "00:00";
  document.getElementById("best-accuracy").textContent = "Acurácia: -";

  // Configuração para enviar
  const config = {
    pop_size: popSize,
    geracoes: numGenerations,
    taxa_mutacao: mutationRate,
    metodo_selecao: selectionMethod,
    elite_size: elitismSize,
    tamanho_torneio: tournamentSize,
    n_epochs: numEpochs,
    space: space,
  };

  startTime = new Date();

  // Função para atualizar o tempo decorrido
  function updateElapsedTime() {
    const elapsedMs = new Date() - startTime;
    const elapsedMinutes = Math.floor(elapsedMs / 60000);
    const elapsedSeconds = Math.floor((elapsedMs % 60000) / 1000);
    document.getElementById("elapsed-time").textContent = `${elapsedMinutes
      .toString()
      .padStart(2, "0")}:${elapsedSeconds.toString().padStart(2, "0")}`;
  }

  // Iniciar o cronômetro
  elapsedInterval = setInterval(updateElapsedTime, 1000);
  updateElapsedTime();

  // Enviar requisição para iniciar o AG
  fetch("http://localhost:8000/start_ag", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
      "Access-Control-Request-Headers": "content-type",
    },
    body: JSON.stringify(config),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.task_id) {
        currentTaskId = data.task_id;
        startLogPolling(currentTaskId);
        showToast(
          "O algoritmo iniciou! Acompanhe o progresso no dashboard.",
          "success",
          3000
        );

        // Iniciar verificação de status
        intervalId = setInterval(
          () => checkTaskStatus(currentTaskId, numGenerations),
          2000
        );

        // Configurar listener do botão de parar
        document.getElementById("stop-ag").onclick = async function () {
          try {
            const response = await fetch(
              `http://localhost:8000/stop_task/${currentTaskId}`,
              {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  "ngrok-skip-browser-warning": "true",
                },
              }
            );

            if (response.ok) {
              stopAlgorithm();
              showToast("Execução parada com sucesso", "success", 3000);
            } else {
              throw new Error("Falha ao parar a tarefa no servidor");
            }
          } catch (error) {
            stopAlgorithm();
            showToast(
              "Execução interrompida (pode haver tarefas pendentes no servidor)",
              "warning",
              3000
            );
          }
        };
      } else {
        throw new Error("Falha ao iniciar o AG: " + JSON.stringify(data));
      }
    })
    .catch((error) => {
      stopAlgorithm(false);
      showToast("Erro ao iniciar: " + error.message, "error", 5000);
    });
}

// Função startLogPolling atualizada para limpar o intervalo corretamente
function startLogPolling(taskId) {
  // Inicializa o armazenamento de logs para esta task
  task_logs[taskId] = task_logs[taskId] || [];

  if (logIntervalId) {
    clearInterval(logIntervalId);
  }

  logIntervalId = setInterval(async () => {
    if (!agRunning) {
      clearInterval(logIntervalId);
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/logs/${taskId}`, {
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
      });
      const data = await response.json();

      // Atualiza os logs locais
      task_logs[taskId] = data.logs;

      const logConsole = document.getElementById("log-console");
      logConsole.textContent = task_logs[taskId].join("\n");
      logConsole.scrollTop = logConsole.scrollHeight;

      // Analisar logs para encontrar a melhor acurácia
      task_logs[taskId].forEach((logEntry) => {
        const accuracyMatch = logEntry.match(/Acurácia: (\d+\.\d+)/);
        if (accuracyMatch) {
          const currentAccuracy = parseFloat(accuracyMatch[1]);
          if (currentAccuracy > overallBestAccuracy) {
            overallBestAccuracy = currentAccuracy;
            document.getElementById("overall-best-accuracy").textContent =
              (overallBestAccuracy * 100).toFixed(2) + "%";
          }
        }
      });
    } catch (error) {
      console.error("Erro ao buscar logs:", error);
    }
  }, 1000);
}

function checkTaskStatus(taskId, totalGenerations) {
  fetch(`http://localhost:8000/status/${taskId}`, {
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
      "Access-Control-Request-Headers": "content-type",
    },
  })
    .then((response) => response.json())
    .then((statusData) => {
      if (!statusData.result) return;

      const progress = statusData.progress;
      document.getElementById("progress-bar").style.width = `${progress}%`;
      document.getElementById("progress-text").textContent = `${Math.round(
        (progress * totalGenerations) / 100
      )}/${totalGenerations}`;

      // Atualizar estatísticas de tempo
      const elapsedMs = new Date() - startTime;
      const elapsedMinutes = Math.floor(elapsedMs / 60000);
      const elapsedSeconds = Math.floor((elapsedMs % 60000) / 1000);
      document.getElementById("elapsed-time").textContent = `${elapsedMinutes
        .toString()
        .padStart(2, "0")}:${elapsedSeconds.toString().padStart(2, "0")}`;

      if (statusData.status === "completed") {
        clearInterval(intervalId);
        clearInterval(logIntervalId);
        clearInterval(elapsedInterval);
        updateClassificationExamples(taskId);
        resetBestIndividual();
        agRunning = false;
        document.getElementById("stop-ag").disabled = true;
        document.getElementById("stop-ag").style.display = "none";
        document.getElementById("start-ag").disabled = false;
        document.getElementById("start-ag").innerHTML =
          '<i class="bi bi-play-fill"></i> Iniciar Algoritmo';

        // Atualizar o melhor indivíduo
        const bestIndividual = statusData.result.best_individual;
        if (bestIndividual) {
          showBestIndividual(bestIndividual);
          const finalAccuracy = bestIndividual.accuracy * 100;
          document.getElementById("overall-best-accuracy").textContent =
            finalAccuracy.toFixed(2) + "%";
          overallBestAccuracy = bestIndividual.accuracy;

          // Armazenar resultados no histórico
          const resultData = {
            timestamp: Date.now(),
            elapsedTime: document.getElementById("elapsed-time").textContent,
            bestAccuracy: bestIndividual.accuracy,
            bestIndividual: bestIndividual,
            config: {
              popSize: parseInt(document.getElementById("pop-size").value),
              numGenerations: parseInt(
                document.getElementById("num-generations").value
              ),
              numEpochs: parseInt(document.getElementById("num-epochs").value),
              mutationRate: parseFloat(
                document.getElementById("mutation-rate").value
              ),
              selectionMethod:
                document.getElementById("selection-method").value,
              elitismSize: parseInt(
                document.getElementById("elitism-size").value
              ),
              tournamentSize: parseInt(
                document.getElementById("tournament-size").value
              ),
            },
            logs: task_logs[taskId] || [],
            chartData: {
              labels: fitnessChart.data.labels,
              bestAccuracy: fitnessChart.data.datasets[0].data,
              avgAccuracy: fitnessChart.data.datasets[1].data,
            },
          };

          storeResultInHistory(resultData);
        }

        // Atualizar gráfico com o histórico
        const history = statusData.result.history;
        if (history) {
          updateChartWithHistory(history, totalGenerations);
        }
      } else if (statusData.status === "failed") {
        clearInterval(intervalId);
        clearInterval(logIntervalId);
        agRunning = false;
        document.getElementById("start-ag").disabled = false;
        document.getElementById("start-ag").innerHTML =
          '<i class="bi bi-play-fill"></i> Iniciar Algoritmo';
        showToast(
          "O algoritmo falhou: " +
            (statusData.result?.error || "Erro desconhecido"),
          "error",
          5000
        );
      }
    })
    .catch((error) => {
      console.error("Erro ao verificar status:", error);
    });
}

function updateChartWithHistory(history, totalGenerations) {
  if (!history || history.length === 0) return;

  // Limpar dados anteriores
  generationData = [];
  fitnessChart.data.labels = [];
  fitnessChart.data.datasets[0].data = [];
  fitnessChart.data.datasets[1].data = [];

  // Adicionar dados de cada geração
  for (let i = 0; i < history.length; i++) {
    const genData = history[i];
    if (genData.length > 0) {
      const bestAccuracy = Math.max(...genData);
      const avgAccuracy = genData.reduce((a, b) => a + b, 0) / genData.length;

      generationData.push({
        generation: i + 1,
        bestAccuracy: bestAccuracy,
        avgAccuracy: avgAccuracy,
      });

      fitnessChart.data.labels.push(`Geração ${i + 1}`);
      fitnessChart.data.datasets[0].data.push(bestAccuracy);
      fitnessChart.data.datasets[1].data.push(avgAccuracy);
    }
  }

  // Ajustar escala Y
  const maxAccuracy = Math.max(...fitnessChart.data.datasets[0].data);
  fitnessChart.options.scales.y.max = Math.min(maxAccuracy + 0.1, 1);
  fitnessChart.update();
}

function showBestIndividual(individual) {
  const bestIndividualDiv = document.getElementById("best-individual");
  const bestAccuracySpan = document.getElementById("best-accuracy");

  bestAccuracySpan.textContent = `Acurácia: ${(
    individual.accuracy * 100
  ).toFixed(2)}%`;

  let html = '<h5 class="text-center mb-3">Parâmetros do Melhor Indivíduo</h5>';
  html += '<div class="row">';

  for (const [key, value] of Object.entries(individual.params)) {
    const formattedKey = key
      .replace("_", " ")
      .replace(/\b\w/g, (l) => l.toUpperCase());
    let formattedValue = value;

    if (key === "learning_rate") {
      formattedValue = parseFloat(value).toExponential(2);
    } else if (key === "weight_decay") {
      formattedValue = value === 0 ? "0" : parseFloat(value).toExponential(2);
    }

    html += `
              <div class="col-md-6 mb-2">
                  <div class="d-flex justify-content-between">
                      <span class="fw-bold">${formattedKey}:</span>
                      <span class="text-muted">${formattedValue}</span>
                  </div>
              </div>
          `;
  }

  html += "</div>";
  bestIndividualDiv.innerHTML = html;
}

function resetBestIndividual() {
  const bestIndividualDiv = document.getElementById("best-individual");
  bestIndividualDiv.innerHTML = "";
}

// Adicione estas funções ao index.js

// Armazenar resultados no localStorage
function storeResultInHistory(resultData) {
  let history = JSON.parse(localStorage.getItem("agResultsHistory")) || [];
  history.push(resultData);
  localStorage.setItem("agResultsHistory", JSON.stringify(history));
  updateResultsTable();
}

// Atualizar a tabela de resultados
function updateResultsTable() {
  const history = JSON.parse(localStorage.getItem("agResultsHistory")) || [];
  const tableBody = document.getElementById("results-table");
  tableBody.innerHTML = "";

  history.forEach((result, index) => {
    const row = document.createElement("tr");

    row.innerHTML = `
      <td class="text-center">${index + 1}</td>
      <td class="text-center">${result.elapsedTime}</td>
      <td class="text-center">${(result.bestAccuracy * 100).toFixed(2)}%</td>
      <td class="text-center">${formatDate(result.timestamp)}</td>
      <td>
        <button class="btn btn-sm btn-outline-dark view-individual" data-index="${index}">
          <i class="bi bi-eye"></i> Visualizar
        </button>
      </td>
    `;

    tableBody.appendChild(row);
  });

  // Adicionar event listeners aos botões
  document.querySelectorAll(".view-individual").forEach((button) => {
    button.addEventListener("click", function () {
      const index = parseInt(this.getAttribute("data-index"));
      const history =
        JSON.parse(localStorage.getItem("agResultsHistory")) || [];
      if (index >= 0 && index < history.length) {
        showIndividualDetails(history[index], index);
      } else {
        console.error("Índice inválido:", index);
      }
    });
  });
}
// Mostrar detalhes de um indivíduo em um modal
function showIndividualDetails(result, index) {
  // Criar um modal temporário para mostrar os detalhes
  const modalHTML = `
      <div class="modal fade" id="individualModal" tabindex="-1" aria-hidden="true">
          <div class="modal-dialog modal-lg">
              <div class="modal-content">
                  <div class="modal-header">
                      <h5 class="modal-title">Detalhes da Execução</h5>
                      <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                  </div>
                  <div class="modal-body">
                      <div class="row mb-3">
                          <div class="col-md-6">
                              <div class="card">
                                  <div class="card-header">Estatísticas</div>
                                  <div class="card-body">
                                      <p><strong>Melhor Acurácia:</strong> ${(
                                        result.bestAccuracy * 100
                                      ).toFixed(2)}%</p>
                                      <p><strong>Tempo Decorrido:</strong> ${
                                        result.elapsedTime
                                      }</p>
                                      <p><strong>Data:</strong> ${new Date(
                                        result.timestamp
                                      ).toLocaleString()}</p>
                                  </div>
                              </div>
                          </div>
                          <div class="col-md-6">
                              <div class="card">
                                  <div class="card-header">Configuração do AG</div>
                                  <div class="card-body">
                                      <p><strong>População:</strong> ${
                                        result.config.popSize
                                      }</p>
                                      <p><strong>Gerações:</strong> ${
                                        result.config.numGenerations
                                      }</p>
                                      <p><strong>Método:</strong> ${
                                        result.config.selectionMethod
                                      }</p>
                                      <p><strong>Taxa Mutação:</strong> ${
                                        result.config.mutationRate
                                      }</p>
                                  </div>
                              </div>
                          </div>
                      </div>
                      
                      <div class="card mb-3">
                          <div class="card-header">Melhor Indivíduo</div>
                          <div class="card-body">
                              <div class="row">
                                  ${Object.entries(result.bestIndividual.params)
                                    .map(
                                      ([key, value]) => `
                                      <div class="col-md-6 mb-2">
                                          <strong>${key.replace(
                                            "_",
                                            " "
                                          )}:</strong> ${value}
                                      </div>
                                  `
                                    )
                                    .join("")}
                              </div>
                          </div>
                      </div>
                      
                      <div class="card">
                          <div class="card-header">Logs</div>
                          <div class="card-body" style="max-height: 200px; overflow-y: auto;">
                              <pre style="white-space: pre-wrap;">${result.logs.join(
                                "\n"
                              )}</pre>
                          </div>
                      </div>
                  </div>
                  <div class="modal-footer">
                      <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Fechar</button>
                        <button type="button" class="btn btn-primary" onclick="generatePDF(${index})"> <i class="bi bi-download"></i> Baixar Resultado</button>
                  </div>
              </div>
          </div>
      </div>
  `;

  // Adicionar o modal ao DOM
  document.body.insertAdjacentHTML("beforeend", modalHTML);

  // Mostrar o modal
  const modal = new bootstrap.Modal(document.getElementById("individualModal"));
  modal.show();

  // Remover o modal quando fechado
  document
    .getElementById("individualModal")
    .addEventListener("hidden.bs.modal", function () {
      this.remove();
    });
}

function formatDate(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleDateString() + " " + date.toLocaleTimeString();
}

// Gerar PDF
function generatePDF(resultIndex) {
  const history = JSON.parse(localStorage.getItem("agResultsHistory"));
  const result = history[resultIndex];

  // Criar um novo jsPDF instance
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();

  // Adicionar título
  doc.setFontSize(20);
  doc.text("Resultados do Algoritmo Genético", 105, 15, null, null, "center");

  // Adicionar data
  doc.setFontSize(10);
  doc.text(
    `Gerado em: ${new Date().toLocaleString()}`,
    105,
    22,
    null,
    null,
    "center"
  );

  // Adicionar linha divisória
  doc.line(20, 25, 190, 25);

  // Adicionar informações básicas
  doc.setFontSize(14);
  doc.text("Informações da Execução", 20, 35);

  doc.setFontSize(12);
  doc.text(
    `Data da execução: ${new Date(result.timestamp).toLocaleString()}`,
    20,
    45
  );
  doc.text(
    `Melhor acurácia: ${(result.bestAccuracy * 100).toFixed(2)}%`,
    20,
    55
  );
  doc.text(`Tempo decorrido: ${result.elapsedTime}`, 20, 65);

  // Adicionar configurações do AG
  doc.setFontSize(14);
  doc.text("Configurações do Algoritmo Genético", 20, 80);

  doc.setFontSize(12);
  let y = 90;
  Object.entries(result.config).forEach(([key, value]) => {
    doc.text(`${key}: ${value}`, 20, y);
    y += 10;
  });

  // Adicionar melhor indivíduo
  doc.setFontSize(14);
  doc.text("Melhor Indivíduo", 20, y + 15);

  doc.setFontSize(12);
  y += 25;
  Object.entries(result.bestIndividual.params).forEach(([key, value]) => {
    doc.text(`${key.replace("_", " ")}: ${value}`, 20, y);
    y += 10;
  });

  // Adicionar logs (em uma nova página)
  doc.addPage();
  doc.setFontSize(14);
  doc.text("Logs da Execução", 105, 15, null, null, "center");

  doc.setFontSize(10);
  const logs = result.logs.join("\n");
  doc.text(logs, 20, 25, {
    maxWidth: 170,
    align: "left",
  });

  // Salvar o PDF
  doc.save(`resultado_ag_${resultIndex + 1}.pdf`);
}
