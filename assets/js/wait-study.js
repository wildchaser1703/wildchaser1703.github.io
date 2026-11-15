const trace = [
  { time: 1000, word: 'The' },
  { time: 2000, word: 'wait' },
  { time: 3000, word: 'is' },
  { time: 4000, word: 'over.' }
];
const output = document.querySelector('[data-output]');
const status = document.querySelector('[data-status]');
const marks = [...document.querySelectorAll('[data-trace] span')];
const stop = document.querySelector('[data-stop]');
let timers = [];

function reset() {
  timers.forEach(clearTimeout);
  timers = [];
  stop.disabled = true;
}

function run(mode) {
  reset();
  output.textContent = '';
  marks.forEach(mark => mark.classList.remove('visible'));
  stop.disabled = false;
  status.textContent = mode === 'all' ? 'Running: show at once.' : 'Running: show as ready.';
  trace.forEach((event, index) => {
    const visibleAt = mode === 'all' ? trace[trace.length - 1].time : event.time;
    timers.push(setTimeout(() => {
      output.textContent += `${index ? ' ' : ''}${event.word}`;
      marks[index].classList.add('visible');
      if (index === trace.length - 1) {
        reset();
        status.textContent = 'Complete. Try the other mode or compare the timings below.';
      }
    }, visibleAt));
  });
}

document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => run(button.dataset.mode)));
stop.addEventListener('click', () => {
  reset();
  output.textContent = '';
  marks.forEach(mark => mark.classList.remove('visible'));
  status.textContent = 'Stopped. Choose a mode to start again.';
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && timers.length) stop.click();
});
