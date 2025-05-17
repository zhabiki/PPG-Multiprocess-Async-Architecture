const fs = require('fs');
const csv = require('csv-parser');
const axios = require('axios');

// Путь к CSV
const csvFilePath = '01_exp02_anxiety.csv';

// Массив для хранения значений
const ledValues = [];

// Считываем CSV
fs.createReadStream(csvFilePath)
  .pipe(csv())
  .on('data', (row) => {
    if (row['afe_LED1ABSVAL']) {
      ledValues.push(Number(row['afe_LED1ABSVAL']));
    }
  })
  .on('end', async () => {
    console.log(`Прочитано ${ledValues.length} значений. Отправка...`);

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        data: ledValues
      });
      console.log('Ответ от FastAPI:', response.data);
    } catch (error) {
      console.error('Ошибка при POST-запросе: ', error.message);
      console.log(' --  ТАК ЧТО ХУЙ ТЕБЕ!!!')
    }
  });
