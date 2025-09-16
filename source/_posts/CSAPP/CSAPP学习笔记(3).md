---
title: CSAPP学习笔记(2)
date: 2025-04-28 10:15:53
tags: CSAPP
categories: 学习
mathjax: true
---
# 虚拟地址

## 物理和虚拟寻址

计算机系统的主存被组织成一个由M个连续的字节大小的单元组成的数组。每字节都有一个唯一的物理地址（PA）。第一个字节的地址为0，接下来的字节地址为1，再下一个为2，依此类推。给定这种简单的结构，CPU访问内存的最自然的方式就是使用物理地址。我们把这种方式称为物理寻址。

早期的PC采用物理寻址，现代处理器使用的是一种虚拟寻址的寻址方式。

![](https://fastly.jsdelivr.net/gh/lazysqrt2/blog-img/20250428112317630.png)

使用虚拟寻址，CPU通过生成一个虚拟地址（VirtualAddress，VA)来访问主存，这个虚拟地址在被送到内存之前先转换成适当的物理地址。将一个虚拟地址转换为物理地址的任务叫做地址翻译（addresstranslation)。就像异常处理一样，地址翻译需要CPU硬件和操作系统之间的紧密合作。CPU芯片上叫做内存管理单元（MemoryManagementUnit，MMU)的专用硬件，利用存放在主存中的查询表来动态翻译虚拟地址，该表的内容由操作
系统管理。

## 虚拟内存的三个角色

### 作为缓存工具

概念上来说，虚拟内存就是存储在磁盘上的 N 个连续字节的数组。这个数组的部分内容，会缓存在 DRAM 中，在 DRAM 中的每个缓存块(cache block)就称为页(page)

![](https://fastly.jsdelivr.net/gh/lazysqrt2/blog-img/20250428113824690.png)

大致的思路和之前的 cache memory 是类似的，就是利用 DRAM 比较快的特性，把最常用的数据换缓存起来。如果要访问磁盘的话，大约会比访问 DRAM 慢一万倍，所以我们的目标就是尽可能从 DRAM 中拿数据。为此，我们需要：

* 更大的页尺寸(page size)：通常是 4KB，有的时候可以达到 4MB
* 全相联(Fully associative)：每一个虚拟页(virual page)可以放在任意的物理页(physical page)中，没有限制。
* 映射函数非常复杂，所以没有办法用硬件实现，通常使用 Write-back 而非 Write-through 机制
  * Write-through: 命中后更新缓存，同时写入到内存中
  * Write-back: 直到这个缓存需要被置换出去，才写入到内存中（需要额外的 dirty bit 来表示缓存中的数据是否和内存中相同，因为可能在其他的时候内存中对应地址的数据已经更新，那么重复写入就会导致原有数据丢失）

每个进程都有一个页表，页表负责将虚拟页映射到物理页。操作系统负责维护页表的内容，以及在磁盘和DRAM之间来回传送页。

![](https://fastly.jsdelivr.net/gh/lazysqrt2/blog-img/20250428114319241.png)

当查询到页表中已经缓存了的地址，可以直接在DRAM中访问对应数据。

不命中的时候，即访问到 page table 中灰色条目的时候，因为在 DRAM 中并没有对应的数据，所以需要执行一系列操作（从磁盘复制到 DRAM 中），具体为：

* 触发 Page fault，也就是一个异常
* Page fault handler 会选择 DRAM 中需要被置换的 page，并把数据从磁盘复制到 DRAM 中
* 重新执行访问指令，这时候就会是 page hit

复制过程中的等待时间称为 demand paging。

仔细留意上面的页表，会发现有一个条目是 null，也就是没有分配。具体的分配过程（比方说声明了一个大数组），就是让该条目指向虚拟内存（在磁盘上）的某个页，但并不复制到 DRAM，只有当出现 page fault 的时候才需要拷贝数据。


[参考文献](https://wdxtub.com/csapp/thin-csapp-1/2016/04/16/)（巨佬 Orz）注：本文有部分内容是直接复制来源于原文，目的仅用于学习
