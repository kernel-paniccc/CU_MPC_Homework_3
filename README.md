# distributed-lab

Лаборатория для демонстрации p2p и collective-операций в torch.distributed.

Для запуска:
1. Указать в `docker-compose.yml` в разделе `command` для каждого воркера нужную функцию.
2. Выполнить `docker compose up`.

После внесения изменений в код:
1. Остановить и удалить прерыдущие контейнеры: 
```bash
docker compose down
```

2. Запустить, пересобрав новые образы:
```bash
docker compose up --build
```