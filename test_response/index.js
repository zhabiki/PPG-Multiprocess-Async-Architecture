const fs = require('fs');
const csv = require('csv-parser');
const readline = require('readline');
const axios = require('axios');
const { resolve } = require('path');

async function example_1() {
  const ledValues = [];
  fs.createReadStream('../lib/dataproc/examples/maus_006_ppg_pixart_resting.csv')
    .pipe(csv())
    .on('data', (row) => {
      if (row['Resting']) {
        ledValues.push(Number(row['Resting']));
      }
    })
    .on('end', async () => {
      console.time('Итоговое время примера №1');
      console.log(`\n=== Пример №1 -- Из датасета MAUS (${ledValues.length} зн.) ===`);
      try {
        const response = await axios.post(
          'http://localhost:8000/predict', {
            "fs": 100,
            "data": ledValues
          }
        );
        console.log('\nОтвет от FastAPI для примера №1:', response.data);
      } catch (error) {
        console.error('\nОшибка при отправке POST-запроса:', error.message);
      }
      resolve();
      console.timeEnd('Итоговое время примера №1');
    });
}

async function example_2() {
  const ledValues = [];
  readline.createInterface({
      input: fs.createReadStream('../lib/dataproc/examples/250409-Н-315-120.txt'),
      crlfDelay: Infinity
    })
    .on('line', (line) => {
      const num = Number(line.trim());
      if (!isNaN(num)) {
        ledValues.push(num);
      }
    })
    .on('close', async () => {
      console.time('Итоговое время примера №2');
      console.log(`\n=== Пример №2 -- ФПГ с пальца (${ledValues.length} зн.) ===`);
      try {
        const response = await axios.post(
          'http://localhost:8000/predict',
          {
            fs: 120,
            data: ledValues
          }
        );
        console.log('\nОтвет от FastAPI для примера №2:', response.data);
      } catch (error) {
        console.error('\nОшибка при отправке POST-запроса:', error.message);
      }
      resolve();
      console.timeEnd('Итоговое время примера №2');
    });
}

async function example_3() {
  const ledValues = [];
  fs.createReadStream('../lib/dataproc/examples/01_exp02_anxiety.csv')
    .pipe(csv())
    .on('data', (row) => {
      if (row['afe_LED1ABSVAL']) {
        ledValues.push(Number(row['afe_LED1ABSVAL']));
      }
    })
    .on('end', async () => {
      console.time('Итоговое время примера №3');
      console.log(`\n=== Пример №3 -- ФПГ с запястья (${ledValues.length} зн.) ===`);
      try {
        const response = await axios.post(
          'http://localhost:8000/predict', {
            "fs": 250,
            "data": ledValues
          }
        );
        console.log('\nОтвет от FastAPI для примера №3:', response.data);
      } catch (error) {
        console.error('\nОшибка при отправке POST-запроса:', error.message);
      }
      resolve();
      console.timeEnd('Итоговое время примера №3');
    });
}

// Запускаем примеры ПОСЛЕДОВАТЕЛЬНО, иначе капец будет
(async () => {
  await example_1();
  await example_2();
  await example_3();
})();
