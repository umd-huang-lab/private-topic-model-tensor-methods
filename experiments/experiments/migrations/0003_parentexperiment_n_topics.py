# Generated by Django 3.0.7 on 2020-06-13 20:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('experiments', '0002_auto_20200604_2203'),
    ]

    operations = [
        migrations.AddField(
            model_name='parentexperiment',
            name='n_topics',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
