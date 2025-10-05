"""
CLI скрипт для предсказания DGA доменов.
"""

import argparse
import sys
import json
from pathlib import Path

from dga_classifier.predictor import FastDGAPredictor
from dga_classifier.utils import setup_logging


def predict_single_domain(model_path: str, domain: str, log_level: str = "INFO"):
    """
    Предсказание для одного домена.
    """
    logger = setup_logging(log_level=log_level, console_output=True)
    
    try:
        predictor = FastDGAPredictor(model_path=model_path)
        result = predictor.predict_single(domain)
        
        print(f"\n🔍 Анализ домена: {domain}")
        print(f"📊 Результат: {result['prediction_label']}")
        print(f"📈 Вероятность DGA: {result['dga_probability']:.4f}")
        print(f"🎯 Уверенность: {result['confidence']:.4f}")
        print(f"⚡ Время инференса: {result['inference_time_ms']:.2f}мс")
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        return None


def predict_batch_domains(
    model_path: str, 
    domains: list, 
    batch_size: int = 1000,
    n_jobs: int = None,
    log_level: str = "INFO"
):
    """
    Предсказание для списка доменов.
    """
    logger = setup_logging(log_level=log_level, console_output=True)
    
    try:
        predictor = FastDGAPredictor(model_path=model_path)
        results = predictor.predict_batch(domains, batch_size=batch_size, n_jobs=n_jobs)
        
        # Статистика
        dga_count = sum(1 for r in results if r['prediction'] == 1)
        legit_count = len(results) - dga_count
        avg_inference_time = sum(r['inference_time_ms'] for r in results) / len(results)
        
        print(f"\n📊 Статистика предсказаний:")
        print(f"   Всего доменов: {len(results)}")
        print(f"   🔴 DGA: {dga_count}")
        print(f"   🟢 Legit: {legit_count}")
        print(f"   ⚡ Среднее время: {avg_inference_time:.2f}мс/домен")
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        return None


def predict_from_file(
    model_path: str,
    input_file: str,
    output_file: str = None,
    batch_size: int = 10000,
    n_jobs: int = None,
    log_level: str = "INFO"
):
    """
    Предсказание для файла с доменами.
    """
    logger = setup_logging(log_level=log_level, console_output=True)
    
    if output_file is None:
        output_file = f"{Path(input_file).stem}_predictions.csv"
    
    try:
        predictor = FastDGAPredictor(model_path=model_path)
        predictor.predict_file(
            input_file=input_file,
            output_file=output_file,
            batch_size=batch_size,
            n_jobs=n_jobs
        )
        
        print(f"✅ Предсказания сохранены в: {output_file}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")


def benchmark_performance(
    model_path: str,
    test_domains: list = None,
    n_runs: int = 3,
    log_level: str = "INFO"
):
    """
    Бенчмарк производительности модели.
    """
    logger = setup_logging(log_level=log_level, console_output=True)
    
    if test_domains is None:
        test_domains = [
            "google.com", "facebook.com", "amazon.com", "microsoft.com",
            "kjahsdkjahsd.com", "asdkjhaslkjdh.net", "qwertyasdfgh.org",
            "randomdomain123.net", "testdomain456.com", "example789.org"
        ] * 100  # 1000 доменов для теста
    
    try:
        predictor = FastDGAPredictor(model_path=model_path)
        stats = predictor.get_performance_stats(test_domains, n_runs=n_runs)
        
        print(f"\n🚀 Бенчмарк производительности:")
        print(f"   Доменов в тесте: {stats['domains_count']:,}")
        print(f"   Прогонов: {n_runs}")
        print(f"   Среднее время: {stats['avg_total_time_sec']:.3f}с")
        print(f"   Время на домен: {stats['avg_time_per_domain_ms']:.2f}мс")
        print(f"   Пропускная способность: {stats['throughput_domains_per_sec']:.0f} доменов/с")
        print(f"   Мин./макс. время: {stats['min_time_sec']:.3f}с / {stats['max_time_sec']:.3f}с")
        
        return stats
        
    except Exception as e:
        logger.error(f"Ошибка при бенчмарке: {e}")
        return None


def main():
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(
        description="Предсказание DGA доменов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "model_path",
        help="Путь к файлу модели (.joblib)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")
    
    # Команда для одного домена
    single_parser = subparsers.add_parser("single", help="Предсказание для одного домена")
    single_parser.add_argument("domain", help="Доменное имя")
    
    # Команда для списка доменов
    batch_parser = subparsers.add_parser("batch", help="Предсказание для списка доменов")
    batch_parser.add_argument("domains", nargs="+", help="Список доменов")
    batch_parser.add_argument("--batch-size", type=int, default=1000, help="Размер батча")
    batch_parser.add_argument("--n-jobs", type=int, help="Количество процессов")
    
    # Команда для файла
    file_parser = subparsers.add_parser("file", help="Предсказание для файла с доменами")
    file_parser.add_argument("input_file", help="Входной файл (по одному домену на строку)")
    file_parser.add_argument("--output-file", "-o", help="Выходной файл (по умолчанию auto)")
    file_parser.add_argument("--batch-size", type=int, default=10000, help="Размер батча")
    file_parser.add_argument("--n-jobs", type=int, help="Количество процессов")
    
    # Команда для бенчмарка
    bench_parser = subparsers.add_parser("benchmark", help="Бенчмарк производительности")
    bench_parser.add_argument("--n-runs", type=int, default=3, help="Количество прогонов")
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Уровень логирования"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Проверяем модель
    if not Path(args.model_path).exists():
        print(f"❌ Файл модели не найден: {args.model_path}")
        sys.exit(1)
    
    # Выполняем команду
    if args.command == "single":
        result = predict_single_domain(args.model_path, args.domain, args.log_level)
        sys.exit(0 if result else 1)
        
    elif args.command == "batch":
        results = predict_batch_domains(
            args.model_path, 
            args.domains,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            log_level=args.log_level
        )
        
        # Выводим результаты
        if results:
            print(f"\n📝 Подробные результаты:")
            for result in results[:10]:  # Показываем первые 10
                print(f"   {result['domain']}: {result['prediction_label']} "
                      f"(p={result['dga_probability']:.3f})")
            
            if len(results) > 10:
                print(f"   ... и еще {len(results) - 10} доменов")
        
        sys.exit(0 if results else 1)
        
    elif args.command == "file":
        predict_from_file(
            args.model_path,
            args.input_file,
            output_file=args.output_file,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            log_level=args.log_level
        )
        sys.exit(0)
        
    elif args.command == "benchmark":
        stats = benchmark_performance(
            args.model_path,
            n_runs=args.n_runs,
            log_level=args.log_level
        )
        sys.exit(0 if stats else 1)


if __name__ == "__main__":
    main()