import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import styles from './DqnPlayground.module.css';
import {
  RL_BASIC_CHAPTERS,
  getAdjacentRlBasicChapters,
  getReadingModeHref,
  getRlBasicChapter,
} from '../interactiveRoutes';

const ACTIONS = [
  {index: 0, key: 'left', label: '左移', short: 'L'},
  {index: 1, key: 'right', label: '右移', short: 'R'},
];

const PAGE_NOTES = [
  {
    id: 'network',
    step: '01',
    title: 'Q 网络与探索',
    summary: 'DQN 用网络近似 Q(s, a)，再用 ε-greedy 在探索与利用之间做平衡。',
    bullets: [
      '这个沙盘把“网络输出”直接展开成 5 个离散状态上的 Q 值，便于观察数值如何变化。',
      '当随机数小于 ε 时，智能体会探索；否则就执行当前 Q 值最大的动作。',
    ],
    callout: '先把 ε 调高，连续点几次“单步交互”，你会看到智能体在还没学会之前更愿意乱试。',
  },
  {
    id: 'replay',
    step: '02',
    title: '经验回放',
    summary: 'DQN 不会立刻拿最新样本更新网络，而是先写进 replay buffer，再随机采样小批量训练。',
    bullets: [
      '随机采样可以打散时间相关性，让训练数据更接近独立同分布。',
      '同一条样本可以被多次复用，提高与环境交互得到的数据价值。',
    ],
    callout: '留意 replay buffer 中被高亮的卡片，它们就是本轮训练真正参与更新的样本。',
  },
  {
    id: 'target',
    step: '03',
    title: '目标网络',
    summary: '目标网络让 bootstrap 的目标值暂时固定，避免在线网络一边预测一边移动终点。',
    bullets: [
      '右侧两张 Q 表分别是 online network 和 target network 的输出。',
      '只有当优化步数达到同步间隔时，target network 才会整体拷贝 online network。',
    ],
    callout: '把同步间隔调大一些，再连续训练几轮，你会明显看到两张表出现“滞后差”。',
  },
  {
    id: 'console',
    step: '04',
    title: '控制台',
    summary: '在最后一页直接调整 ε、γ、mini-batch 和目标同步间隔，并观察 TD 目标与损失如何变化。',
    bullets: [
      '终止状态的目标值就是即时奖励；非终止状态还要加上 γ × max Q_target(s\').',
      '调完参数后，看右侧公式卡片里最近一个 mini-batch 的 target、TD error 和均方损失如何一起变化。',
    ],
    callout: '可以先在前三页理解机制，再来到最后一页把所有超参数联动起来观察。',
  },
];

const START_STATE = 2;
const GOAL_STATE = 4;
const BUFFER_CAPACITY = 14;
const CURRENT_DOC_ID = 'rl_basic/ch7/README';

function createQTable() {
  return Array.from({length: 5}, () => [0, 0]);
}

function cloneQTable(table) {
  return table.map((row) => [...row]);
}

function createSimulationState() {
  return {
    currentState: START_STATE,
    onlineQ: createQTable(),
    targetQ: createQTable(),
    replayBuffer: [],
    envStepCount: 0,
    optimizeCount: 0,
    episodeCount: 1,
    lastDecision: null,
    lastTransition: null,
    lastBatch: null,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function pickGreedyAction(qValues) {
  if (qValues[0] === qValues[1]) {
    return 1;
  }

  return qValues[1] > qValues[0] ? 1 : 0;
}

function stepEnvironment(state, actionIndex) {
  const delta = actionIndex === 0 ? -1 : 1;
  const nextState = clamp(state + delta, 0, GOAL_STATE);

  if (nextState === GOAL_STATE) {
    return {nextState, reward: 1, done: true};
  }

  if (state === 0 && actionIndex === 0) {
    return {nextState, reward: -0.35, done: false};
  }

  return {nextState, reward: -0.04, done: false};
}

function sampleBatchIndices(length, batchSize) {
  const pool = Array.from({length}, (_, index) => index);

  for (let index = pool.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    const temp = pool[index];
    pool[index] = pool[swapIndex];
    pool[swapIndex] = temp;
  }

  return pool.slice(0, batchSize);
}

function formatNumber(value) {
  const fixed = value.toFixed(2);
  return value > 0 ? `+${fixed}` : fixed;
}

function formatAction(actionIndex) {
  return ACTIONS[actionIndex].label;
}

function classNames(...values) {
  return values.filter(Boolean).join(' ');
}

function SliderControl({label, value, min, max, step, onChange, hint}) {
  return (
    <label className={styles.controlCard}>
      <div className={styles.controlHeader}>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <input
        className={styles.slider}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
      />
      <span className={styles.controlHint}>{hint}</span>
    </label>
  );
}

function NetworkTable({title, table, currentState, sampledStates, tint, maxAbsValue}) {
  return (
    <section className={styles.networkCard}>
      <div className={styles.networkHeader}>
        <h3>{title}</h3>
        <span className={styles.networkTint} style={{'--tint-color': tint}}>
          输出层快照
        </span>
      </div>
      <div className={styles.table}>
        {table.map((row, stateIndex) => (
          <div
            key={`${title}-${stateIndex}`}
            className={classNames(
              styles.tableRow,
              stateIndex === currentState && styles.tableRowCurrent,
              sampledStates.has(stateIndex) && styles.tableRowSampled,
            )}
          >
            <div className={styles.stateLabel}>
              <span>S{stateIndex}</span>
              {stateIndex === GOAL_STATE ? <em>goal</em> : null}
            </div>
            <div className={styles.actionValues}>
              {row.map((value, actionIndex) => (
                <div key={`${title}-${stateIndex}-${actionIndex}`} className={styles.valueCell}>
                  <span className={styles.valueMeta}>{ACTIONS[actionIndex].short}</span>
                  <div className={styles.valueTrack}>
                    <div
                      className={classNames(
                        styles.valueFill,
                        value >= 0 ? styles.valuePositive : styles.valueNegative,
                      )}
                      style={{
                        width: `${(Math.abs(value) / maxAbsValue) * 100}%`,
                        '--tint-color': tint,
                      }}
                    />
                  </div>
                  <span className={styles.valueNumber}>{formatNumber(value)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function DqnPlayground() {
  const [currentPage, setCurrentPage] = useState(0);
  const [epsilon, setEpsilon] = useState(0.28);
  const [gamma, setGamma] = useState(0.9);
  const [learningRate, setLearningRate] = useState(0.34);
  const [batchSize, setBatchSize] = useState(4);
  const [syncInterval, setSyncInterval] = useState(4);
  const [sim, setSim] = useState(createSimulationState);

  const currentNote = PAGE_NOTES[currentPage];
  const isConsolePage = currentPage === PAGE_NOTES.length - 1;
  const currentChapter = getRlBasicChapter(CURRENT_DOC_ID);
  const {previousChapter, nextChapter} = getAdjacentRlBasicChapters(CURRENT_DOC_ID);
  const sampledStates = new Set(
    sim.lastBatch ? sim.lastBatch.samples.map((sample) => sample.state) : [],
  );
  const sampledStepIds = new Set(
    sim.lastBatch ? sim.lastBatch.samples.map((sample) => sample.stepId) : [],
  );
  const flattenedValues = [...sim.onlineQ.flat(), ...sim.targetQ.flat(), 1, -1];
  const maxAbsValue = Math.max(...flattenedValues.map((value) => Math.abs(value)));
  const syncCountdown = syncInterval - (sim.optimizeCount % syncInterval || syncInterval);
  const averageGap =
    sim.onlineQ.flat().reduce((gap, value, index) => {
      const targetValue = sim.targetQ.flat()[index];
      return gap + Math.abs(value - targetValue);
    }, 0) / sim.onlineQ.flat().length;
  const focusSample = sim.lastBatch ? sim.lastBatch.samples[0] : null;

  const runSteps = (count) => {
    let nextSim = {
      ...sim,
      onlineQ: cloneQTable(sim.onlineQ),
      targetQ: cloneQTable(sim.targetQ),
      replayBuffer: sim.replayBuffer.map((transition) => ({...transition})),
    };

    for (let step = 0; step < count; step += 1) {
      const qValues = nextSim.onlineQ[nextSim.currentState];
      const greedyAction = pickGreedyAction(qValues);
      const roll = Math.random();
      const action = roll < epsilon ? Math.floor(Math.random() * ACTIONS.length) : greedyAction;
      const transitionResult = stepEnvironment(nextSim.currentState, action);

      nextSim.envStepCount += 1;
      nextSim.lastDecision = {
        mode: roll < epsilon ? 'explore' : 'exploit',
        roll,
        action,
        greedyAction,
      };
      nextSim.lastTransition = {
        stepId: nextSim.envStepCount,
        state: nextSim.currentState,
        action,
        reward: transitionResult.reward,
        nextState: transitionResult.nextState,
        done: transitionResult.done,
      };

      nextSim.replayBuffer = [
        ...nextSim.replayBuffer,
        nextSim.lastTransition,
      ].slice(-BUFFER_CAPACITY);

      if (nextSim.replayBuffer.length >= batchSize) {
        const indices = sampleBatchIndices(nextSim.replayBuffer.length, batchSize);
        const sampledTransitions = indices.map((index) => nextSim.replayBuffer[index]);
        const onlineSnapshot = cloneQTable(nextSim.onlineQ);
        const updatedOnline = cloneQTable(nextSim.onlineQ);
        let meanLoss = 0;

        const samples = sampledTransitions.map((sample) => {
          const nextTargetValues = nextSim.targetQ[sample.nextState];
          const target = sample.done
            ? sample.reward
            : sample.reward + gamma * Math.max(...nextTargetValues);
          const predictionBefore = updatedOnline[sample.state][sample.action];
          const tdError = target - predictionBefore;
          const loss = tdError * tdError;
          meanLoss += loss;
          updatedOnline[sample.state][sample.action] =
            predictionBefore + learningRate * tdError;

          return {
            ...sample,
            onlineSnapshot: onlineSnapshot[sample.state][sample.action],
            predictionBefore,
            predictionAfter: updatedOnline[sample.state][sample.action],
            nextTargetValues: [...nextTargetValues],
            target,
            tdError,
            loss,
          };
        });

        nextSim.onlineQ = updatedOnline;
        nextSim.optimizeCount += 1;
        const synced = nextSim.optimizeCount % syncInterval === 0;

        if (synced) {
          nextSim.targetQ = cloneQTable(nextSim.onlineQ);
        }

        nextSim.lastBatch = {
          optimizeCount: nextSim.optimizeCount,
          synced,
          meanLoss: meanLoss / samples.length,
          samples,
        };
      }

      if (transitionResult.done) {
        nextSim.currentState = START_STATE;
        nextSim.episodeCount += 1;
      } else {
        nextSim.currentState = transitionResult.nextState;
      }
    }

    setSim(nextSim);
  };

  const syncTargetNow = () => {
    setSim((current) => ({
      ...current,
      targetQ: cloneQTable(current.onlineQ),
      lastBatch: current.lastBatch
        ? {
            ...current.lastBatch,
            synced: true,
          }
        : current.lastBatch,
    }));
  };

  const resetSimulation = () => {
    setSim(createSimulationState());
  };

  const lessonPanel = (
    <section className={styles.lessonCard}>
      <span className={styles.lessonStep}>{currentNote.step}</span>
      <h2>{currentNote.title}</h2>
      <p className={styles.lessonSummary}>{currentNote.summary}</p>
      <ul className={styles.lessonList}>
        {currentNote.bullets.map((bullet) => (
          <li key={bullet}>{bullet}</li>
        ))}
      </ul>
      <div className={styles.callout}>
        <strong>操作提示</strong>
        <p>{currentNote.callout}</p>
      </div>
    </section>
  );

  const controlPanel = (
    <section className={styles.controlPanel}>
      <div className={styles.panelTitle}>
        <span className={styles.lessonStep}>{currentNote.step}</span>
        <h3>训练控制台</h3>
        <span>{currentNote.summary}</span>
      </div>

      <ul className={styles.controlList}>
        {currentNote.bullets.map((bullet) => (
          <li key={bullet}>{bullet}</li>
        ))}
      </ul>

      <div className={styles.controlGrid}>
        <SliderControl
          label="ε 探索率"
          value={epsilon.toFixed(2)}
          min={0.05}
          max={0.95}
          step={0.01}
          onChange={(event) => setEpsilon(Number(event.target.value))}
          hint="越高越愿意随机探索"
        />
        <SliderControl
          label="γ 折扣因子"
          value={gamma.toFixed(2)}
          min={0.5}
          max={0.99}
          step={0.01}
          onChange={(event) => setGamma(Number(event.target.value))}
          hint="越高越重视未来回报"
        />
        <SliderControl
          label="学习率 α"
          value={learningRate.toFixed(2)}
          min={0.05}
          max={0.6}
          step={0.01}
          onChange={(event) => setLearningRate(Number(event.target.value))}
          hint="控制 online network 追目标值的速度"
        />
        <SliderControl
          label="mini-batch"
          value={String(batchSize)}
          min={2}
          max={6}
          step={1}
          onChange={(event) => setBatchSize(Number(event.target.value))}
          hint="每次从 replay buffer 随机抽取多少条样本"
        />
        <SliderControl
          label="目标同步间隔 C"
          value={String(syncInterval)}
          min={2}
          max={8}
          step={1}
          onChange={(event) => setSyncInterval(Number(event.target.value))}
          hint="每经过多少次优化，把 online network 拷贝给 target network"
        />
      </div>

      <div className={styles.actionRow}>
        <button type="button" className={styles.primaryButton} onClick={() => runSteps(1)}>
          单步交互
        </button>
        <button type="button" className={styles.secondaryButton} onClick={() => runSteps(10)}>
          连续 10 步
        </button>
        <button type="button" className={styles.secondaryButton} onClick={syncTargetNow}>
          立即同步目标网络
        </button>
        <button type="button" className={styles.ghostButton} onClick={resetSimulation}>
          重置
        </button>
      </div>

      <div className={styles.callout}>
        <strong>操作提示</strong>
        <p>{currentNote.callout}</p>
      </div>
    </section>
  );

  const environmentCard = (
    <section className={styles.labCard}>
      <div className={styles.cardHeader}>
        <div>
          <h3>交互走廊环境</h3>
          <p>从中间状态出发，向右抵达 goal 会获得 +1 奖励。</p>
        </div>
        {sim.lastDecision ? (
          <span
            className={classNames(
              styles.modeBadge,
              sim.lastDecision.mode === 'explore'
                ? styles.modeExplore
                : styles.modeExploit,
            )}
          >
            {sim.lastDecision.mode === 'explore' ? '探索中' : '利用中'}
          </span>
        ) : (
          <span className={styles.modeBadge}>等待第一步</span>
        )}
      </div>

      <div className={styles.corridor}>
        {Array.from({length: 5}, (_, index) => (
          <div
            key={`state-${index}`}
            className={classNames(
              styles.stateNode,
              index === sim.currentState && styles.stateNodeCurrent,
              index === GOAL_STATE && styles.stateNodeGoal,
              sampledStates.has(index) && styles.stateNodeSampled,
            )}
          >
            <span className={styles.stateName}>S{index}</span>
            <span className={styles.stateHint}>
              {index === GOAL_STATE ? 'goal' : index === START_STATE ? 'start' : 'state'}
            </span>
          </div>
        ))}
      </div>

      <div className={styles.transitionStrip}>
        <div>
          <span>当前位置</span>
          <strong>S{sim.currentState}</strong>
        </div>
        <div>
          <span>最近动作</span>
          <strong>
            {sim.lastDecision ? formatAction(sim.lastDecision.action) : '尚未执行'}
          </strong>
        </div>
        <div>
          <span>随机数 / ε</span>
          <strong>
            {sim.lastDecision
              ? `${sim.lastDecision.roll.toFixed(2)} / ${epsilon.toFixed(2)}`
              : '--'}
          </strong>
        </div>
        <div>
          <span>下次自动同步</span>
          <strong>
            {sim.optimizeCount === 0 ? syncInterval : syncCountdown || syncInterval} 次优化后
          </strong>
        </div>
      </div>
    </section>
  );

  const replayCard = (
    <section className={styles.labCard}>
      <div className={styles.cardHeader}>
        <div>
          <h3>Replay Buffer</h3>
          <p>容量固定为 {BUFFER_CAPACITY}，最新样本会顶掉最旧样本。</p>
        </div>
        <span className={styles.counterBadge}>
          {sim.replayBuffer.length} / {BUFFER_CAPACITY}
        </span>
      </div>

      <div className={styles.bufferList}>
        {sim.replayBuffer.length === 0 ? (
          <div className={styles.emptyState}>
            先执行几次交互，右侧就会开始积累 transition。
          </div>
        ) : (
          [...sim.replayBuffer].reverse().map((transition) => (
            <article
              key={`transition-${transition.stepId}`}
              className={classNames(
                styles.bufferCard,
                sampledStepIds.has(transition.stepId) && styles.bufferCardActive,
              )}
            >
              <header>
                <span>step {transition.stepId}</span>
                <strong>{sampledStepIds.has(transition.stepId) ? '本轮采样' : '缓冲中'}</strong>
              </header>
              <div className={styles.bufferBody}>
                <span>S{transition.state}</span>
                <em>{formatAction(transition.action)}</em>
                <span>S{transition.nextState}</span>
              </div>
              <footer>
                <span>r = {formatNumber(transition.reward)}</span>
                <span>{transition.done ? 'terminal' : 'continue'}</span>
              </footer>
            </article>
          ))
        )}
      </div>
    </section>
  );

  const formulaCard = (
    <section className={styles.labCard}>
      <div className={styles.cardHeader}>
        <div>
          <h3>Bellman Target 拆解</h3>
          <p>观察最近一次 mini-batch 中的第一条样本如何生成 TD 目标。</p>
        </div>
        <span className={styles.counterBadge}>
          {sim.lastBatch ? `batch ${sim.lastBatch.optimizeCount}` : '等待采样'}
        </span>
      </div>

      {focusSample ? (
        <div className={styles.formulaStack}>
          <div className={styles.formulaCard}>
            <span className={styles.formulaLabel}>样本</span>
            <strong>
              (S{focusSample.state}, {formatAction(focusSample.action)}, r = {formatNumber(focusSample.reward)}, S{focusSample.nextState})
            </strong>
          </div>
          <div className={styles.formulaCard}>
            <span className={styles.formulaLabel}>目标值 y</span>
            <code>
              {focusSample.done
                ? `y = r = ${formatNumber(focusSample.reward)}`
                : `y = r + γ max Q_target(s') = ${formatNumber(focusSample.reward)} + ${gamma.toFixed(2)} × max(${focusSample.nextTargetValues
                    .map((value) => formatNumber(value))
                    .join(', ')}) = ${formatNumber(focusSample.target)}`}
            </code>
          </div>
          <div className={styles.formulaCard}>
            <span className={styles.formulaLabel}>TD 误差</span>
            <code>
              δ = y - Q_online(s, a) = {formatNumber(focusSample.target)} - {formatNumber(focusSample.predictionBefore)} = {formatNumber(focusSample.tdError)}
            </code>
          </div>
          <div className={styles.formulaCard}>
            <span className={styles.formulaLabel}>更新后</span>
            <code>
              Q ← {formatNumber(focusSample.predictionAfter)}，当前 batch MSE = {sim.lastBatch.meanLoss.toFixed(3)}
            </code>
          </div>
          <div className={styles.batchList}>
            {sim.lastBatch.samples.map((sample) => (
              <div key={`sample-${sample.stepId}`} className={styles.batchItem}>
                <span>
                  step {sample.stepId}: S{sample.state} / {formatAction(sample.action)}
                </span>
                <strong>{formatNumber(sample.tdError)}</strong>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className={styles.emptyState}>
          Replay buffer 里至少要有 {batchSize} 条样本，系统才会开始随机采样并计算 TD target。
        </div>
      )}
    </section>
  );

  const targetInsightCard = (
    <section className={styles.labCard}>
      <div className={styles.cardHeader}>
        <div>
          <h3>目标网络节奏</h3>
          <p>这一页只看 online 和 target 之间的节奏差。</p>
        </div>
        <span className={styles.counterBadge}>
          {sim.lastBatch?.synced ? '刚刚同步' : '尚未同步'}
        </span>
      </div>
      <div className={styles.insightGrid}>
        <div className={styles.insightItem}>
          <span>优化步数</span>
          <strong>{sim.optimizeCount}</strong>
        </div>
        <div className={styles.insightItem}>
          <span>平均参数差</span>
          <strong>{averageGap.toFixed(2)}</strong>
        </div>
        <div className={styles.insightItem}>
          <span>下一次硬同步</span>
          <strong>
            {sim.optimizeCount === 0 ? syncInterval : syncCountdown || syncInterval} 步后
          </strong>
        </div>
      </div>
    </section>
  );

  let visualStage = null;
  if (currentPage === 0) {
    visualStage = (
      <div className={styles.visualStageSingle}>
        {environmentCard}
        <NetworkTable
          title="Online Network"
          table={sim.onlineQ}
          currentState={sim.currentState}
          sampledStates={sampledStates}
          tint="#0f766e"
          maxAbsValue={maxAbsValue}
        />
      </div>
    );
  } else if (currentPage === 1) {
    visualStage = <div className={styles.visualStageSingle}>{replayCard}</div>;
  } else if (currentPage === 2) {
    visualStage = (
      <div className={styles.visualStageSingle}>
        <div className={styles.networkGrid}>
          <NetworkTable
            title="Online Network"
            table={sim.onlineQ}
            currentState={sim.currentState}
            sampledStates={sampledStates}
            tint="#0f766e"
            maxAbsValue={maxAbsValue}
          />
          <NetworkTable
            title="Target Network"
            table={sim.targetQ}
            currentState={sim.currentState}
            sampledStates={sampledStates}
            tint="#b45309"
            maxAbsValue={maxAbsValue}
          />
        </div>
        {targetInsightCard}
      </div>
    );
  } else {
    visualStage = (
      <div className={styles.visualStageSingle}>
        {formulaCard}
        {environmentCard}
      </div>
    );
  }

  return (
    <div className={styles.shell}>
      <div className={styles.toolbar}>
        <div className={styles.toolbarSide}>
          <Link
            className={styles.toolbarLink}
            to={getReadingModeHref(currentChapter?.readingHref ?? '/rl_basic/ch7/')}
          >
            阅读模式
          </Link>
        </div>

        <div className={styles.toolbarCenter}>
          <details className={styles.chapterSwitcher}>
            <summary className={styles.chapterSummary}>
              <span>{currentChapter?.navLabel ?? 'DQN 算法'}</span>
              <span className={styles.chapterSummaryArrow} aria-hidden="true">
                ▾
              </span>
            </summary>
            <div className={styles.chapterMenu}>
              {RL_BASIC_CHAPTERS.map((chapter) => (
                <Link
                  key={chapter.docId}
                  className={classNames(
                    styles.chapterMenuItem,
                    chapter.docId === CURRENT_DOC_ID && styles.chapterMenuItemActive,
                  )}
                  to={chapter.preferredHref}
                  aria-current={chapter.docId === CURRENT_DOC_ID ? 'page' : undefined}
                >
                  <span>{chapter.navLabel}</span>
                  {chapter.hasInteractive ? (
                    <strong className={styles.chapterModeBadge}>交互</strong>
                  ) : (
                    <strong className={styles.chapterModeBadgeMuted}>阅读</strong>
                  )}
                </Link>
              ))}
            </div>
          </details>
        </div>

        <div className={classNames(styles.toolbarSide, styles.toolbarPager)}>
          {previousChapter ? (
            <Link className={styles.toolbarLink} to={previousChapter.preferredHref}>
              上一章
            </Link>
          ) : (
            <span className={classNames(styles.toolbarLink, styles.toolbarLinkDisabled)}>
              上一章
            </span>
          )}
          {nextChapter ? (
            <Link className={styles.toolbarLink} to={nextChapter.preferredHref}>
              下一章
            </Link>
          ) : (
            <span className={classNames(styles.toolbarLink, styles.toolbarLinkDisabled)}>
              下一章
            </span>
          )}
        </div>
      </div>

      <aside className={styles.sidebar}>
        <div className={styles.sidebarTopNav}>
          <button
            type="button"
            className={styles.navButton}
            disabled={currentPage === 0}
            onClick={() => setCurrentPage((page) => Math.max(0, page - 1))}
          >
            上一页
          </button>

          <label className={styles.pageSelectWrap}>
            <select
              className={styles.pageSelect}
              value={currentPage}
              onChange={(event) => setCurrentPage(Number(event.target.value))}
              aria-label="切换当前交互页"
            >
              {PAGE_NOTES.map((page, index) => (
                <option key={page.id} value={index}>
                  {page.step} · {page.title}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className={styles.navButton}
            disabled={currentPage === PAGE_NOTES.length - 1}
            onClick={() => setCurrentPage((page) => Math.min(PAGE_NOTES.length - 1, page + 1))}
          >
            下一页
          </button>
        </div>

        <div className={styles.sidebarStage}>
          {currentPage === PAGE_NOTES.length - 1 ? controlPanel : lessonPanel}
        </div>
      </aside>

      <section className={styles.visualPanel}>
        <div className={styles.heroCompact}>
          <div className={styles.metricGrid}>
            <div className={styles.metricCard}>
              <span>环境步数</span>
              <strong>{sim.envStepCount}</strong>
            </div>
            <div className={styles.metricCard}>
              <span>Episode</span>
              <strong>{sim.episodeCount}</strong>
            </div>
            <div className={styles.metricCard}>
              <span>优化步数</span>
              <strong>{sim.optimizeCount}</strong>
            </div>
            <div className={styles.metricCard}>
              <span>Online/Target 差值</span>
              <strong>{averageGap.toFixed(2)}</strong>
            </div>
          </div>
        </div>

        <div className={styles.visualStage}>{visualStage}</div>
      </section>
    </div>
  );
}
